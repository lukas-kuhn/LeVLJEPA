import datetime
import os
import random
import math

import hydra
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchvision.ops import MLP
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model, AutoTokenizer
from tqdm import tqdm
import wandb
from huggingface_hub import sync_bucket

from utils.dataset import CC12MMulti
from utils.eval_utils import run_imagenet_eval, ViTAllTokens
from utils.sigreg import SIGReg


class GatherLayer(torch.autograd.Function):
    """All-gather with gradients flowing back through each rank's slice."""

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


@hydra.main(version_base=None, config_path="configs", config_name="levljepa_plus")
def main(cfg: DictConfig):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    torch.cuda.set_device(local_rank)
    DEVICE = f"cuda:{local_rank}"

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    run_name = cfg.run_name
    if is_main:
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    BATCH_SIZE = cfg.batch_size
    TOTAL_STEPS = cfg.total_steps
    LR = cfg.lr
    LAMBDA_VISION = cfg.lambda_vision
    LAMBDA_TEXT = cfg.lambda_text
    LAMBDA_MULTI = cfg.lambda_multi
    WARMUP_STEPS = cfg.warmup_steps
    ETA_MIN = cfg.eta_min
    MAX_GRAD_NORM = cfg.max_grad_norm
    HIDDEN_SIZE = cfg.model.hidden_size
    EMBED_DIM = cfg.model.embed_dim
    N_GLOBAL = cfg.global_crops_number
    N_LOCAL = cfg.local_crops_number
    N_VIEWS = N_GLOBAL + N_LOCAL

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_samples = cfg.get("max_samples", None)
    dataset = CC12MMulti(
        local_crops_number=N_LOCAL, tokenizer=tokenizer, max_samples=max_samples
    )
    if is_main:
        print(f"[dataset] CC12M size: {len(dataset):,}")
        if max_samples is None:
            assert len(dataset) > 1_000_000
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=True,
        num_workers=8,
        persistent_workers=False,
        pin_memory=True,
        timeout=120,
    )

    text_base = GPT2Model(
        GPT2Config(
            n_embd=HIDDEN_SIZE,
            n_layer=cfg.model.num_layers,
            n_head=cfg.model.num_heads,
            n_inner=HIDDEN_SIZE * 4,
            vocab_size=tokenizer.vocab_size,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
        )
    ).to(DEVICE)

    vision_base = timm.create_model(
        cfg.model.vit, pretrained=False, num_classes=0, dynamic_img_size=True
    ).to(DEVICE)

    # Linear projection applied to the LayerNormed CLS token before SIGReg.
    # SIGReg requires variance that LayerNorm suppresses; this projection restores it.
    vision_pre_proj = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 2048),
        nn.BatchNorm1d(2048),
        nn.GELU(),
        nn.Linear(2048, EMBED_DIM),
    ).to(DEVICE)

    text_pre_proj = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, 2048),
        nn.BatchNorm1d(2048),
        nn.GELU(),
        nn.Linear(2048, EMBED_DIM),
    ).to(DEVICE)

    proj_hidden = [cfg.projector_width] * cfg.projector_depth + [EMBED_DIM]
    projector_vision = MLP(
        EMBED_DIM, proj_hidden, norm_layer=nn.BatchNorm1d, activation_layer=nn.GELU
    ).to(DEVICE)
    projector_text = MLP(
        EMBED_DIM, proj_hidden, norm_layer=nn.BatchNorm1d, activation_layer=nn.GELU
    ).to(DEVICE)

    vision_pre_proj = nn.SyncBatchNorm.convert_sync_batchnorm(vision_pre_proj)
    text_pre_proj = nn.SyncBatchNorm.convert_sync_batchnorm(text_pre_proj)
    projector_vision = nn.SyncBatchNorm.convert_sync_batchnorm(projector_vision)
    projector_text = nn.SyncBatchNorm.convert_sync_batchnorm(projector_text)

    text_base = DDP(text_base, device_ids=[local_rank])
    vision_base = DDP(vision_base, device_ids=[local_rank])
    vision_pre_proj = DDP(vision_pre_proj, device_ids=[local_rank])
    text_pre_proj = DDP(text_pre_proj, device_ids=[local_rank])
    projector_vision = DDP(projector_vision, device_ids=[local_rank])
    projector_text = DDP(projector_text, device_ids=[local_rank])

    sigreg = SIGReg().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(text_base.parameters())
                + list(text_pre_proj.parameters()),
                "lr": LR,
            },
            {
                "params": list(vision_base.parameters())
                + list(vision_pre_proj.parameters()),
                "lr": LR,
            },
            {"params": projector_vision.parameters(), "lr": LR},
            {"params": projector_text.parameters(), "lr": LR},
        ]
    )

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return ETA_MIN / LR + (1 - ETA_MIN / LR) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        text_base.train()
        vision_base.train()

        epoch = 0
        sampler.set_epoch(epoch)
        data_iter = iter(dataloader)

        with tqdm(range(TOTAL_STEPS), desc="Training", disable=not is_main) as t:
            for global_step in t:
                try:
                    global_crops, local_crops, text_inputs = next(data_iter)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    data_iter = iter(dataloader)
                    global_crops, local_crops, text_inputs = next(data_iter)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    global_crops = rearrange(
                        global_crops.to(DEVICE), "b v c h w -> (v b) c h w"
                    )
                    local_crops = rearrange(
                        local_crops.to(DEVICE), "b v c h w -> (v b) c h w"
                    )
                    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

                    global_raw = vision_base(global_crops)
                    local_raw = vision_base(local_crops)
                    all_raw = torch.cat([global_raw, local_raw], dim=0)
                    all_pre = vision_pre_proj(all_raw)
                    all_pre = rearrange(all_pre, "(v b) d -> v b d", v=N_VIEWS)

                    text_raw = text_base(**text_inputs).last_hidden_state[:, -1, :]
                    text_linear = text_pre_proj(text_raw)

                    sigreg_vision = sigreg(all_pre)
                    sigreg_text = sigreg(text_linear)

                    vision_mean = all_pre[:N_GLOBAL].mean(dim=0)
                    mse_multi = (all_pre - vision_mean.unsqueeze(0)).square().mean()

                    all_pre_flat = rearrange(all_pre, "v b d -> (v b) d")
                    all_image_proj = projector_vision(all_pre_flat)
                    text_proj = projector_text(text_linear)

                    mse_loss_cross_text = (
                        (text_linear.repeat(N_VIEWS, 1).detach() - all_image_proj).square().mean()
                    )
                    mse_loss_cross_vision = (
                        (vision_mean.detach() - text_proj).square().mean()
                    )
                    image_proj = all_image_proj[:len(text_linear)]

                    mse_cross = (mse_loss_cross_text + mse_loss_cross_vision) / 2

                    loss = (
                        (1 - (LAMBDA_VISION + LAMBDA_TEXT + LAMBDA_MULTI)) * mse_cross
                        + LAMBDA_VISION * sigreg_vision
                        + LAMBDA_TEXT * sigreg_text
                        + LAMBDA_MULTI * mse_multi
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                all_params = [p for g in optimizer.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(all_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()

                if is_main:
                    with torch.no_grad():
                        image_linear = all_pre[0]

                        v_proj = F.normalize(image_proj.float(), dim=-1)
                        t_proj = F.normalize(text_linear.float(), dim=-1)
                        logits_proj = v_proj @ t_proj.T / 0.07
                        pos_logits_proj = logits_proj.diagonal().mean()
                        lse_proj = torch.logsumexp(logits_proj, dim=1).mean()

                        v_base = F.normalize(image_linear.float(), dim=-1)
                        t_base = F.normalize(text_linear.float(), dim=-1)
                        logits_base = v_base @ t_base.T / 0.07
                        pos_logits_base = logits_base.diagonal().mean()
                        lse_base = torch.logsumexp(logits_base, dim=1).mean()

                        cos_sim_matched = F.cosine_similarity(
                            image_linear, text_linear
                        ).mean()
                        cos_sim_random = F.cosine_similarity(
                            image_linear, text_linear.roll(1, 0)
                        ).mean()
                        cos_sim_img_pred = F.cosine_similarity(
                            image_proj, text_linear
                        ).mean()
                        cos_sim_img_pred_random = F.cosine_similarity(
                            image_proj, text_linear.roll(1, 0)
                        ).mean()
                        cos_sim_txt_pred = F.cosine_similarity(
                            text_proj, image_linear
                        ).mean()
                        cos_sim_txt_pred_random = F.cosine_similarity(
                            text_proj, image_linear.roll(1, 0)
                        ).mean()

                        img_centered = image_linear - image_linear.mean(0)
                        sv = torch.linalg.svdvals(img_centered.float())
                        sv_norm = sv / sv.sum()
                        effective_rank_vision = torch.exp(
                            -torch.sum(sv_norm * torch.log(sv_norm + 1e-7))
                        )

                        txt_centered = text_linear - text_linear.mean(0)
                        sv_t = torch.linalg.svdvals(txt_centered.float())
                        sv_norm_t = sv_t / sv_t.sum()
                        effective_rank_text = torch.exp(
                            -torch.sum(sv_norm_t * torch.log(sv_norm_t + 1e-7))
                        )

                    metrics = {
                        "pos_logits_proj": pos_logits_proj.item(),
                        "lse_proj": lse_proj.item(),
                        "gap_proj": (pos_logits_proj - lse_proj).item(),
                        "pos_logits_base": pos_logits_base.item(),
                        "lse_base": lse_base.item(),
                        "gap_base": (pos_logits_base - lse_base).item(),
                        "sigreg_vision": sigreg_vision.item(),
                        "sigreg_text": sigreg_text.item(),
                        "mse_loss_cross_text": mse_loss_cross_text.item(),
                        "mse_loss_cross_vision": mse_loss_cross_vision.item(),
                        "mse_multi": mse_multi.item(),
                        "cos_sim_matched": cos_sim_matched.item(),
                        "cos_sim_random": cos_sim_random.item(),
                        "cos_sim_img_pred": cos_sim_img_pred.item(),
                        "cos_sim_img_pred_random": cos_sim_img_pred_random.item(),
                        "cos_sim_txt_pred": cos_sim_txt_pred.item(),
                        "cos_sim_txt_pred_random": cos_sim_txt_pred_random.item(),
                        "effective_rank_vision": effective_rank_vision.item(),
                        "effective_rank_text": effective_rank_text.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    wandb.log(metrics, step=global_step)

                if is_main:
                    t.set_postfix(
                        {
                            "loss": loss.item(),
                            "mse_cross_text": mse_loss_cross_text.item(),
                            "mse_cross_vision": mse_loss_cross_vision.item(),
                            "mse_multi": mse_multi.item(),
                            "sigreg_vision": sigreg_vision.item(),
                        }
                    )

                if (global_step + 1) % cfg.save_every_n_steps == 0:
                    if is_main:
                        text_ckpt = {
                            "encoder": text_base.module.state_dict(),
                            "pre_proj": text_pre_proj.module.state_dict(),
                            "projector": projector_text.module.state_dict(),
                        }
                        vision_ckpt = {
                            "encoder": vision_base.module.state_dict(),
                            "pre_proj": vision_pre_proj.module.state_dict(),
                            "projector": projector_vision.module.state_dict(),
                        }
                        torch.save(
                            text_ckpt,
                            f"{cfg.output_dir}/{cfg.run_name}_text_step{global_step + 1}.pt",
                        )
                        torch.save(
                            vision_ckpt,
                            f"{cfg.output_dir}/{cfg.run_name}_vision_step{global_step + 1}.pt",
                        )
                        sync_bucket(
                            cfg.output_dir,
                            f"hf://buckets/{cfg.hf_bucket}/{cfg.run_name}",
                            include=[f"{cfg.run_name}_*.pt"],
                            delete=True,
                        )
                    dist.barrier()

                if (global_step + 1) % cfg.eval_every_n_steps == 0:
                    if is_main:
                        try:
                            eval_metrics = run_imagenet_eval(
                                vision_model=vision_base.module,
                                text_model=text_base.module,
                                tokenizer=tokenizer,
                                cache_dir=cfg.cache_dir,
                                device=DEVICE,
                                vision_embed=vision_pre_proj.module,
                                text_embed=text_pre_proj.module,
                                proj_vision=projector_vision.module,
                                proj_text=projector_text.module,
                                vision_model_full=ViTAllTokens(vision_base.module),
                                batch_size=cfg.batch_size,
                                num_workers=8,
                                linear_probe=cfg.get("linear_probe", False),
                            )
                            wandb.log({"step": global_step + 1, **eval_metrics})
                        except Exception as e:
                            print(f"[eval] failed with: {e}")
                    dist.barrier()

    finally:
        if is_main:
            wandb.finish()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

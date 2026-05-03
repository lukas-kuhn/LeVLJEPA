import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from tqdm import tqdm


PROMPT_TEMPLATE = "a photo of a {}"


class ViTAllTokens(nn.Module):
    """Wraps a timm ViT to return all tokens (CLS + patches) via forward_features."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit.forward_features(x)


class ImageNetVal(Dataset):
    def __init__(self, cache_dir, split="validation"):
        self.ds = load_dataset("ILSVRC/imagenet-1k", split=split, cache_dir=cache_dir)
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        return self.transform(sample["image"].convert("RGB")), sample["label"]

    @property
    def class_names(self):
        return self.ds.features["label"].names


def _torch_linear_probe(
    feats_np, labels_np, device, num_classes=1000, lr=0.1, epochs=50, batch_size=4096
):
    """GPU-accelerated linear probe on an 80/20 train/test split."""
    n = len(feats_np)
    split = int(0.8 * n)
    X_train = torch.from_numpy(feats_np[:split]).float().to(device)
    y_train = torch.from_numpy(labels_np[:split]).long().to(device)
    X_test = torch.from_numpy(feats_np[split:]).float().to(device)
    y_test = torch.from_numpy(labels_np[split:]).long().to(device)

    mean = X_train.mean(0)
    std = X_train.std(0).clamp(min=1e-8)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    head = nn.Linear(X_train.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        perm = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            loss = F.cross_entropy(head(X_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        top1 = (head(X_test).argmax(1) == y_test).float().mean().item() * 100
    return top1


def _effective_rank(features: torch.Tensor) -> float:
    centered = features - features.mean(0)
    sv = torch.linalg.svdvals(centered.float())
    sv_norm = sv / sv.sum()
    return torch.exp(-torch.sum(sv_norm * torch.log(sv_norm + 1e-7))).item()


def _encode_texts(model, tokenizer, texts, device, batch_size=64, return_full=False):
    all_features = []
    all_masks = []
    token_idx = (
        -1
        if getattr(tokenizer, "pad_token", None)
        == getattr(tokenizer, "eos_token", None)
        else 0
    )
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=77, return_tensors="pt"
        ).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(**inputs).last_hidden_state
            if return_full:
                all_features.append(hidden)
                all_masks.append(inputs["attention_mask"] == 0)
            else:
                all_features.append(hidden[:, token_idx, :])
    if return_full:
        max_len = max(f.size(1) for f in all_features)
        padded_feats = []
        padded_masks = []
        for f, m in zip(all_features, all_masks):
            pad_len = max_len - f.size(1)
            if pad_len > 0:
                f = F.pad(f, (0, 0, 0, pad_len))
                m = F.pad(m, (0, pad_len), value=True)
            padded_feats.append(f)
            padded_masks.append(m)
        return torch.cat(padded_feats, dim=0), torch.cat(padded_masks, dim=0)
    return torch.cat(all_features, dim=0)


def run_imagenet_eval(
    vision_model,
    text_model,
    tokenizer,
    cache_dir,
    device,
    vision_embed=None,
    text_embed=None,
    proj_vision=None,
    proj_text=None,
    pred_vision=None,
    pred_text=None,
    vision_model_full=None,
    batch_size=256,
    num_workers=8,
    linear_probe=False,
    proj_cross_eval=True,
):
    """Zero-shot and optional linear probe eval on ImageNet val.

    All models should be unwrapped (no DDP); they will be set to eval mode
    internally and restored to their original state on return.
    Returns a flat dict of metrics suitable for wandb.log().
    """
    if vision_embed is None:
        vision_embed = nn.Identity()
    if text_embed is None:
        text_embed = nn.Identity()

    was_training = {
        "vision": vision_model.training,
        "text": text_model.training,
        "vision_embed": vision_embed.training,
        "text_embed": text_embed.training,
    }
    vision_model.eval()
    text_model.eval()
    vision_embed.eval()
    text_embed.eval()
    if proj_vision is not None:
        was_training["proj_vision"] = proj_vision.training
        was_training["proj_text"] = proj_text.training
        proj_vision.eval()
        proj_text.eval()
    if pred_vision is not None:
        was_training["pred_vision"] = pred_vision.training
        was_training["pred_text"] = pred_text.training
        pred_vision.eval()
        pred_text.eval()
    if vision_model_full is not None:
        was_training["vision_full"] = vision_model_full.training
        vision_model_full.eval()

    dataset = ImageNetVal(cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    prompts = [
        PROMPT_TEMPLATE.format(name.replace("_", " ")) for name in dataset.class_names
    ]

    metrics = {}

    with torch.no_grad():
        text_features = text_embed(
            _encode_texts(text_model, tokenizer, prompts, device)
        )
        text_features_norm = F.normalize(text_features.float(), dim=-1)
        if proj_text is not None:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                text_proj_norm = F.normalize(proj_text(text_features).float(), dim=-1)

        use_proj = proj_vision is not None
        use_pred = pred_vision is not None
        counts = {"zeroshot": [0, 0]}
        if use_proj:
            counts["proj"] = [0, 0]
            proj_cross = proj_cross_eval and text_proj_norm.shape[-1] == text_features_norm.shape[-1]
            if proj_cross:
                counts["proj_image"] = [0, 0]
                counts["proj_text"] = [0, 0]
                counts["avg"] = [0, 0]
        if use_pred:
            text_full, text_pad_mask = _encode_texts(
                text_model, tokenizer, prompts, device, return_full=True
            )
            text_pred_norm = F.normalize(
                pred_text(text_full, text_pad_mask).float(), dim=-1
            )
            counts["pred_image"] = [0, 0]
            counts["pred_text"] = [0, 0]
            counts["pred_avg"] = [0, 0]

        all_feats, all_feats_mean, all_labels_list = [], [], []
        all_vision_feats_for_rank = []
        total = 0

        for images, labels in tqdm(loader, desc="ImageNet eval"):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_image_features = vision_model(images)
                image_features = vision_embed(raw_image_features)
            image_norm = F.normalize(image_features.float(), dim=-1)
            all_vision_feats_for_rank.append(image_features.float().cpu())

            logits = image_norm @ text_features_norm.T
            counts["zeroshot"][0] += (logits.argmax(dim=-1) == labels).sum().item()
            counts["zeroshot"][1] += (
                (logits.topk(5, dim=-1).indices == labels.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )

            if use_proj:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_proj_norm = F.normalize(
                        proj_vision(image_features).float(), dim=-1
                    )

                logits_proj = image_proj_norm @ text_proj_norm.T
                counts["proj"][0] += (logits_proj.argmax(dim=-1) == labels).sum().item()
                counts["proj"][1] += (
                    (logits_proj.topk(5, dim=-1).indices == labels.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )

                if proj_cross:
                    logits_pi = image_proj_norm @ text_features_norm.T
                    counts["proj_image"][0] += (
                        (logits_pi.argmax(dim=-1) == labels).sum().item()
                    )
                    counts["proj_image"][1] += (
                        (logits_pi.topk(5, dim=-1).indices == labels.unsqueeze(1))
                        .any(dim=1)
                        .sum()
                        .item()
                    )

                    logits_pt = image_norm @ text_proj_norm.T
                    counts["proj_text"][0] += (
                        (logits_pt.argmax(dim=-1) == labels).sum().item()
                    )
                    counts["proj_text"][1] += (
                        (logits_pt.topk(5, dim=-1).indices == labels.unsqueeze(1))
                        .any(dim=1)
                        .sum()
                        .item()
                    )

                    avg_logits = (logits_pi + logits_pt) / 2
                    counts["avg"][0] += (
                        (avg_logits.argmax(dim=-1) == labels).sum().item()
                    )
                    counts["avg"][1] += (
                        (avg_logits.topk(5, dim=-1).indices == labels.unsqueeze(1))
                        .any(dim=1)
                        .sum()
                        .item()
                    )

            if use_pred:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    vision_full_tokens = vision_model_full(images)
                    vision_pred_norm = F.normalize(
                        pred_vision(vision_full_tokens).float(), dim=-1
                    )

                logits_pv = vision_pred_norm @ text_features_norm.T
                counts["pred_image"][0] += (
                    (logits_pv.argmax(dim=-1) == labels).sum().item()
                )
                counts["pred_image"][1] += (
                    (logits_pv.topk(5, dim=-1).indices == labels.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )

                logits_pt2 = image_norm @ text_pred_norm.T
                counts["pred_text"][0] += (
                    (logits_pt2.argmax(dim=-1) == labels).sum().item()
                )
                counts["pred_text"][1] += (
                    (logits_pt2.topk(5, dim=-1).indices == labels.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )

                avg_pred_logits = (logits_pv + logits_pt2) / 2
                counts["pred_avg"][0] += (
                    (avg_pred_logits.argmax(dim=-1) == labels).sum().item()
                )
                counts["pred_avg"][1] += (
                    (avg_pred_logits.topk(5, dim=-1).indices == labels.unsqueeze(1))
                    .any(dim=1)
                    .sum()
                    .item()
                )

            if linear_probe:
                all_feats.append(raw_image_features.float().cpu().numpy())
                if vision_model_full is not None:
                    vision_tokens = (
                        vision_full_tokens if use_pred else vision_model_full(images)
                    )
                    mean_pooled = vision_tokens[:, 1:].mean(dim=1)
                    all_feats_mean.append(mean_pooled.float().cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())

            total += labels.size(0)

    for name, (top1_n, top5_n) in counts.items():
        metrics[f"eval/{name}_top1"] = 100 * top1_n / total
        metrics[f"eval/{name}_top5"] = 100 * top5_n / total

    vision_feats_all = torch.cat(all_vision_feats_for_rank, dim=0)
    metrics["eval/effective_rank_vision"] = _effective_rank(vision_feats_all)
    metrics["eval/effective_rank_text"] = _effective_rank(text_features.float().cpu())

    if linear_probe:
        feats = np.concatenate(all_feats)
        lbls = np.concatenate(all_labels_list)
        metrics["eval/linear_probe_top1"] = _torch_linear_probe(feats, lbls, device)

        if all_feats_mean:
            feats_mean = np.concatenate(all_feats_mean)
            metrics["eval/linear_probe_meanpool_top1"] = _torch_linear_probe(
                feats_mean, lbls, device
            )

    if was_training["vision"]:
        vision_model.train()
    if was_training["text"]:
        text_model.train()
    if was_training["vision_embed"]:
        vision_embed.train()
    if was_training["text_embed"]:
        text_embed.train()
    if proj_vision is not None:
        if was_training["proj_vision"]:
            proj_vision.train()
        if was_training["proj_text"]:
            proj_text.train()
    if pred_vision is not None:
        if was_training["pred_vision"]:
            pred_vision.train()
        if was_training["pred_text"]:
            pred_text.train()
    if vision_model_full is not None:
        if was_training["vision_full"]:
            vision_model_full.train()

    return metrics


def run_imagenet_linear_probe(
    vision_model,
    cache_dir,
    device,
    vision_model_full=None,
    batch_size=256,
    num_workers=8,
):
    """Linear probe eval on ImageNet val (vision-only, no text needed)."""
    was_training = vision_model.training
    vision_model.eval()
    if vision_model_full is not None:
        was_training_full = vision_model_full.training
        vision_model_full.eval()

    dataset = ImageNetVal(cache_dir=cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_feats, all_feats_mean, all_labels_list = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="ImageNet linear probe"):
            images = images.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_features = vision_model(images)
            all_feats.append(image_features.float().cpu().numpy())

            if vision_model_full is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    vision_tokens = vision_model_full(images)
                mean_pooled = vision_tokens[:, 1:].mean(dim=1)
                all_feats_mean.append(mean_pooled.float().cpu().numpy())

            all_labels_list.append(labels.numpy())

    feats = np.concatenate(all_feats)
    lbls = np.concatenate(all_labels_list)
    metrics = {
        "eval/linear_probe_top1": _torch_linear_probe(feats, lbls, device),
    }

    if all_feats_mean:
        feats_mean = np.concatenate(all_feats_mean)
        metrics["eval/linear_probe_meanpool_top1"] = _torch_linear_probe(
            feats_mean, lbls, device
        )

    if was_training:
        vision_model.train()
    if vision_model_full is not None and was_training_full:
        vision_model_full.train()

    return metrics

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from datasets import load_dataset


class DINOAugmentation:
    """DINO-style augmentations with asymmetric global views and Gaussian blur."""

    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        image_size=224,
        local_size=96,
    ):
        self.local_crops_number = local_crops_number

        color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        )
        normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        # Global view 1: always blurred
        self.global_aug1 = T.Compose(
            [
                T.RandomResizedCrop(
                    image_size,
                    scale=global_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0
                ),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )

        # Global view 2: rarely blurred, adds solarization
        self.global_aug2 = T.Compose(
            [
                T.RandomResizedCrop(
                    image_size,
                    scale=global_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1
                ),
                T.RandomSolarize(threshold=128, p=0.2),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )

        # Local crops: blur with p=0.5
        self.local_aug = T.Compose(
            [
                T.RandomResizedCrop(
                    local_size,
                    scale=local_crops_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5
                ),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                normalize,
            ]
        )

    def __call__(self, image):
        global_crops = torch.stack([self.global_aug1(image), self.global_aug2(image)])
        local_crops = torch.stack(
            [self.local_aug(image) for _ in range(self.local_crops_number)]
        )
        return global_crops, local_crops


class CC12MDataset(Dataset):
    def __init__(
        self,
        split="train",
        size=224,
        tokenizer=None,
        max_length=77,
        max_samples=None,
    ):
        self.ds = load_dataset("pixparse/cc12m-wds", split=split)
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize((size, size)),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        for attempt in range(5):
            try:
                item = self.ds[(idx + attempt) % len(self.ds)]
                image = self.transform(item["jpg"].convert("RGB"))
                if self.tokenizer is not None:
                    tokens = self.tokenizer(
                        item["txt"],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    return image, {k: v.squeeze(0) for k, v in tokens.items()}
                return image, item["txt"]
            except Exception:
                continue
        raise RuntimeError(
            f"CC12MDataset: failed to load 5 consecutive samples starting at idx={idx}"
        )


class CC12MMulti(Dataset):
    def __init__(
        self,
        split="train",
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=4,
        image_size=224,
        local_size=96,
        tokenizer=None,
        max_length=77,
        max_samples=None,
        augmentation_type="dino",
    ):
        self.ds = load_dataset("pixparse/cc12m-wds", split=split)
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_transform = T.ToImage()

        self.augment = DINOAugmentation(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            image_size=image_size,
            local_size=local_size,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        for attempt in range(5):
            try:
                item = self.ds[(idx + attempt) % len(self.ds)]
                image = self.base_transform(item["jpg"].convert("RGB"))
                global_crops, local_crops = self.augment(image)
                if self.tokenizer is not None:
                    tokens = self.tokenizer(
                        item["txt"],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    return (
                        global_crops,
                        local_crops,
                        {k: v.squeeze(0) for k, v in tokens.items()},
                    )
                return global_crops, local_crops, item["txt"]
            except Exception:
                continue
        raise RuntimeError(
            f"CC12MMulti: failed to load 5 consecutive samples starting at idx={idx}"
        )

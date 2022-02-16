# generic imports
from typing import Any, Optional, Dict, List
import numpy as np
import logging
import os
import jsonlines
from pathlib import Path
from tqdm.auto import tqdm
from hydra.utils import to_absolute_path

# torch imports
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# custom imports
from bpemb import BPEmb
from sentencepiece import SentencePieceProcessor
from torch.utils.data._utils.collate import default_collate
from i2t.utils import instantiate, ClassDescription


logger = logging.getLogger(__name__)


__all__ = ["I2TDataset"]


def get_image_transform(randomize: bool):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if randomize:
        return transforms.Compose(
            [
                # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )


def text_collate_fn(items):
    ids = []
    offsets = [0]
    for item in items:
        ids.append(torch.tensor(item, dtype=torch.int64))
        offsets.append(len(item))
    return {"ids": torch.cat(ids), "offsets": torch.tensor(offsets[:-1]).cumsum(dim=0)}


class BPEmbTokenizer(BPEmb):
    def __init__(self, model_file: str, **kwargs):
        super().__init__(
            model_file=to_absolute_path(model_file),
            emb_file="unused",
            segmentation_only=True,
            **kwargs
        )

    def __call__(self, text: str) -> List[int]:
        return self.encode_ids(text)


class SentencePieceTokenizer(SentencePieceProcessor):
    def __init__(self, model_file: str, **kwargs):
        super().__init__(model_file=to_absolute_path(model_file), **kwargs)

    def __call__(self, text: str) -> List[int]:
        return self.encode(text, out_type=int)


class I2TDataset(Dataset):
    def __init__(self, data: list, images_directory):

        super().__init__()
        self.data = data
        self.images_directory = Path(images_directory)
        self.randomize = True
        self.image_transform = get_image_transform(randomize=self.randomize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, texts = self.data[idx]
        img = Image.open((self.images_directory / str(img)).with_suffix(".jpg"))

        img = img.convert("RGB")
        img = self.image_transform(img)
        if self.randomize:
            text = np.random.choice(texts)
        else:
            text = texts[0]
        return {"image": img, "text": text}

    @staticmethod
    def collate_fn(items):
        return {
            "image": default_collate([x["image"] for x in items]),
            "text": [x["text"] for x in items],
        }


def get_dataloaders(
    data_train: list,
    images_directory: str,
    batch_size: int,
    dataloader_workers: int,
):
    train_dataset = I2TDataset(data_train, images_directory)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=dataloader_workers,
        drop_last=True,
    )
    return train_dataloader

# generic imports
from typing import Dict, Optional

# torch imports
import torch
from torch import nn

# custom imports
from bpemb import BPEmb
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders import get_encoder
from i2t.utils import instantiate, ClassDescription
from transformers import AutoTokenizer, AutoModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModalityEncoder(nn.Module):
    """Simple wrapper around encoder, adds output projection layer.
    """
    def __init__(
        self,
        encoder: ClassDescription,
        output_dim: int,
        normalize: bool = True
    ):
        super().__init__()
        self.encoder = instantiate(encoder)
        self.projector = nn.Linear(self.encoder.output_dim, output_dim)
        self.normalize = nn.functional.normalize if normalize else (lambda x: x)

    
    def forward(self, *args, **kwargs):
        features = self.encoder(*args, **kwargs)
        projected_features = self.projector(features)
        return self.normalize(projected_features)


class ImageModel(nn.Module):
    """Thin wrapper around SMP encoders.
    """
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        weights: Optional[str] = 'imagenet',
    ):
        super().__init__()
        self.encoder = get_encoder(name=encoder_name, weights=None)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = self.encoder.out_channels[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)[-1]
        x = self.avgpool(x)
        return torch.flatten(x, start_dim=1)


class TextModel(nn.Module):
    """Simple BoW-based text encoder.
    """
    def __init__(
        self,
        model_path: str,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        #model.cuda()
        self.output_dim = 312

    def predict_outputs(self, text):

        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def forward(self, text_data) -> torch.Tensor:
        return torch.cat([self.predict_outputs(text) for text in text_data])

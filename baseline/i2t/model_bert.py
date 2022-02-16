# generic imports
from typing import Dict, Optional

# torch imports
import torch
from torch import nn

# custom imports
from bpemb import BPEmb
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders import get_encoder
from sentence_transformers import SentenceTransformer
from i2t.utils import instantiate, ClassDescription


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
        self.encoder = get_encoder(name=encoder_name, weights=weights)
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
        hidden_size: int = 200,
        hidden_layers: int = 3,
    ):
        super().__init__()

        self.output_dim = 512

        device = "cpu"

        self.sbert = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1', device = device)

        #in_channels = [embedding_size, *(hidden_size for _ in range(hidden_layers))]
        #out_channels = [hidden_size for _ in range(hidden_layers + 1)]
        

    def forward(self, text_data) -> torch.Tensor:
        #x = self.sbert.encode(text_data, convert_to_tensor = True)
        x = self.sbert.encode(text_data, convert_to_tensor = True)
        return x

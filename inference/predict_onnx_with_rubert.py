
# hydra imports

import onnxruntime
import os
from omegaconf import OmegaConf
# generic imports
import time
from typing import Optional, List, Iterable
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import more_itertools

import os
import json
import click

# torch imports
import torch
from torch.utils.data._utils.collate import default_collate

# custom imports
from i2t.system import I2T
from i2t.data_rubert import get_image_transform, text_collate_fn
from i2t.utils import instantiate


class I2TInferer(object):
    def __init__(
        self,
        model_name: str,
        device: str
    ):

        cfg = OmegaConf.load(f'model_cfgs/{model_name}.yaml')

        model = I2T(config=cfg)

        self.ort_session_image = onnxruntime.InferenceSession(f"models/{model_name}/image.onnx")
        self.model_text = model.encoders['text']
        self.model_text.load_state_dict(torch.load(f"models/{model_name}/text_model.pth"))


        self.image_transform = get_image_transform(randomize=False)
        #self.text_collate_fn = text_collate_fn
        self.img_collate_fn = default_collate
        self.device = device
    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
        return self.model_text(texts).numpy()
    
    def encode_images(self, images: Iterable[Image.Image]) -> torch.Tensor:
        pbar = tqdm()
        image_features = []
        for chunk in more_itertools.chunked(images, 10):
            images = [self.image_transform(x.convert('RGB')) for x in chunk]
            images_torch = self.img_collate_fn(images).to(self.device)
            ort_inputs_image = {
                self.ort_session_image.get_inputs()[0].name: self.to_numpy(images_torch),
            }

            chunk_image_features = self.ort_session_image.run(None, ort_inputs_image)[0]
            image_features.append(chunk_image_features)
            pbar.update(len(chunk))
        pbar.close()
        return np.concatenate(image_features, axis=0)
    
    def predict(self, images: Iterable[Image.Image], classes: Iterable[str]) -> np.ndarray:
        text_features = self.encode_texts(classes)
        image_features = self.encode_images(images)
        logits = image_features @ text_features.T
        return np.argmax(logits, axis=1)


@click.command()
@click.option('--model_name', help='Path to inference model')
@click.option('--data_directory', help='Path to directory with evaluation datasets')
@click.option('--predicts_file', help='Path to file where predictions should be put to')
@click.option('--limit_samples', default=None, type=int, help='Limit num of evaluated images')
@click.option('--device', default='cpu', help='PyTorch device')
@click.option('--num_threads', default=None, type=int, help='Optionally force number of torch threads')
@click.option('--dataset', '-d', default=None, multiple=True, help='Optionally select datasets manually')
@torch.no_grad()
def main(
    model_name,
    data_directory: str,
    predicts_file: str,
    limit_samples: Optional[int],
    device: str,
    num_threads: Optional[int],
    dataset: Optional[List[str]]
):
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    inferer = I2TInferer(model_name=model_name, device=device)
    if dataset:
        datasets = dataset
    else:
        datasets = os.listdir(data_directory)

    results = {}
    for dataset in datasets:
        with open(f"{data_directory}/{dataset}/classes.json", 'r') as f:
            classes_labels = json.load(f)
        image_files = os.listdir(f'{data_directory}/{dataset}/img')
        if limit_samples is not None:
            image_files = image_files[:limit_samples]
        images = (Image.open(f'{data_directory}/{dataset}/img/{file}') for file in image_files)
        predicts = inferer.predict(images, classes_labels).tolist()
        predicts = {file.split('.')[0]: predict for file, predict in zip(image_files, predicts)}
        results[dataset] = predicts
    with open(predicts_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)
import torch
import click
from i2t.system import I2T
from i2t.data import get_image_transform, text_collate_fn
from torch.utils.data._utils.collate import default_collate
from i2t.utils import instantiate
from omegaconf import OmegaConf
import os
import onnxruntime
import onnx

ckpt = torch.load("checkpoints/train_emb.ckpt", map_location="cpu")
cfg = OmegaConf.create(ckpt["hyper_parameters"])



def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

# check image

print('check image')
# ort_session_image = onnxruntime.InferenceSession("onnx_models/train_emb_image.onnx")
# image = torch.randn(2, 3, 224, 224, requires_grad=True)

# ort_inputs_image = {
#     ort_session_image.get_inputs()[0].name: to_numpy(image),
# }
# ort_outs_image = ort_session_image.run(None, ort_inputs_image)
# print(ort_outs_image)


# check text
print('check text')
ort_session_text = onnxruntime.InferenceSession("onnx_models/train_emb_text.onnx")

tokenizer = instantiate(cfg._data.tokenizer)


texts = ["русские вперед", "русские вперед !", "три"]
ids = [tokenizer(text) for text in texts]


texts_torch = text_collate_fn(ids)


for inp in ort_session_text.get_inputs():
    print(inp.name)

ort_inputs_text = {
    ort_session_text.get_inputs()[0].name: to_numpy(texts_torch["ids"]),
    ort_session_text.get_inputs()[1].name: to_numpy(texts_torch["offsets"])
}
print(ort_inputs_text)

ort_outs_text = ort_session_text.run(None, ort_inputs_text)
print(ort_outs_text[0])

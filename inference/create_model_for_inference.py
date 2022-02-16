import torch
import click
from i2t.system import I2T
#from i2t.data import get_image_transform, text_collate_fn
from i2t.data_bert import get_image_transform, text_collate_fn
from torch.utils.data._utils.collate import default_collate
from i2t.utils import instantiate
from omegaconf import OmegaConf
import os

@click.command()
@click.option('--inference_ckpt_file')
@click.option('--model_name')
def main(inference_ckpt_file: str, model_name: str):
    out_path = os.path.join('../models_onnx', model_name)
    os.makedirs(out_path)


    ckpt = torch.load(inference_ckpt_file, map_location='cpu')
    cfg = OmegaConf.create(ckpt['hyper_parameters'])
    model = I2T(config=cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    text_model = model.encoders['text']
    image_model = model.encoders['image']
    
    image = torch.randn(1, 3, 224, 224, requires_grad=True)

    
    print('Process text model')
    torch.save(text_model.state_dict(), os.path.join(out_path, 'text_model.pth'))

    
    print('Process image model') 
    #Export the image model
    torch.onnx.export(image_model,  # model being run
        image,  # model input (or a tuple for multiple inputs)
        os.path.join(out_path, 'image.onnx'),
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['image'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={   # variable length axes
                    'image' : {0 : 'batch_size'}})
if __name__ == '__main__':
    main()
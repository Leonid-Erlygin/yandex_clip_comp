

from tqdm.auto import tqdm
import os
import jsonlines
import json
import torch
import numpy as np

def create_new_images_list():
    """
    goes through images in 'metadata.json' file and finds existing in 'images' files
    creates 'metadata_new.json'
    """
    metadata_file = '/home/devel/mlcup_cv/datasets/yandex_images/metadata.json'
    images_directory = '/home/devel/mlcup_cv/datasets/yandex_images/images'
    metadata_new = []
    with jsonlines.open(metadata_file) as reader:
        reader = tqdm(reader)
        for i, obj in enumerate(reader):
            fname = os.path.join(images_directory, str(obj['image']) + '.jpg')
            if os.path.isfile(fname): 
                metadata_new.append(obj)
    with jsonlines.open('metadata_new.json', mode='w') as writer:
        for record in metadata_new:
            writer.write(record)
    

if __name__ == '__main__':
    create_new_images_list()



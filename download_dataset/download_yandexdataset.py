import numpy as np
import os
import cv2
import json
import ast
from multiprocessing import Process

from io import BytesIO
from tqdm import tqdm
import requests
from PIL import Image
from glob import iglob
import warnings
import argparse

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Loader:
    def __init__(
        self,
        process_name,
        init_last_downloaded,
        download_size,
        log_save_period,
        jpg_quality=20,
    ):
        """
        download_size -- number of images that must be dowloaded from images_list
        """
        self.process_name = process_name
        self.image_size = (512, 512)
        self.jpg_quality = jpg_quality
        self.use_gray_scale = False
        self.log_save_period = log_save_period
        self.number_of_fails = 0
        self.number_of_downloaded = 0

        self.path_to_images_list = (
            "/home/devel/mlcup_cv/datasets/yandex_images/images.json"
        )
        self.download_size = download_size

        self.log_file_name = f"processes_logs/last_downloaded_{process_name}"
        if os.path.isfile(self.log_file_name):
            with open(self.log_file_name) as fd:
                self.last_downloaded = int(
                    fd.read()
                )  # индексация списка картинок начинается с 1
        else:
            self.last_downloaded = init_last_downloaded
            with open(self.log_file_name, "w") as fd:
                fd.write(str(init_last_downloaded))

        self.save_dir = (
            f"/home/devel/mlcup_cv/download_dataset/yandex_images/{process_name}"
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def save_log(self, last_downloaded: int):
        with open(self.log_file_name, "w") as fd:
            fd.write(str(last_downloaded))
        print(
            f"Process name {self.process_name}\n Last downloaded {last_downloaded}\n Downloaded {self.number_of_downloaded}\n Fails {self.number_of_fails}"
        )

    def save_one_image(self, image_index, url) -> bool:
        """
        downloads and saves image

        return
        True on success
        """

        try:
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                if image.mode == "P":
                    image = image.convert("RGB")
                image = np.array(image)
                # выкинуть альфа канал, если он есть
                if len(image.shape) != 2 and image.shape[2] == 4:
                    image = image[:, :, :3]

                # странная картинка
                if len(image.shape) != 2 and image.shape[2] == 2:
                    image = image[:, :, 0]

                if len(image.shape) == 2:
                    if image.dtype == "bool":
                        image = (image * 255).astype("uint8")
                    if not self.use_gray_scale:
                        image = np.stack([image, image, image], 2)
                resized = cv2.resize(image, self.image_size)

                if self.use_gray_scale and len(image.shape) == 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    os.path.join(self.save_dir, f"{image_index}.jpg"),
                    resized,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality],
                )
                return True
        except:
            return False
        return False

    def __call__(self):
        with open(self.path_to_images_list) as fd:
            i = 1
            download_count = 0
            for line in fd:
                if i <= self.last_downloaded:
                    i += 1
                    continue
                if i > self.download_size:
                    self.save_log(i - 1)
                    return

                diction = ast.literal_eval(line)
                has_succeded = self.save_one_image(diction["image"], diction["url"])
                download_count += 1
                if download_count % self.log_save_period == 0:
                    self.save_log(i - 1)
                if has_succeded:
                    self.number_of_downloaded += 1
                else:
                    self.number_of_fails += 1
                i += 1

# total 5 462 418 images
# Лёня качает с 1 по 2 000 000 (Скачал. Всего скачано 1 865 847)
# Вова качает с 2 000 000 по 5 000 000
# Отдельно качаем последние 462 418 картинок

if __name__ == "__main__":
    init_last_downloaded = {}
    download_size = {}
    num_processes = 8
    for i in range(num_processes):
        init_last_downloaded[i] = 5000000 + 57802 * i
        download_size[i] = init_last_downloaded[i] + 57802
    processes = []
    print(download_size)
    
    for process_index in range(num_processes):
        p_name = f"process_{process_index}"
        loader = Loader(
            process_name='last_images_' + p_name,
            init_last_downloaded = init_last_downloaded[process_index],
            download_size = download_size[process_index],
            log_save_period = 100
        )
        p = Process(target=loader)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()


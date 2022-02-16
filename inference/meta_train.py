import subprocess
import hydra
from omegaconf import DictConfig

from hydra import compose, initialize
from omegaconf import OmegaConf




def main() -> None:
    initialize(config_path="config", job_name="test_app")
    cfg = compose(config_name="rubert")

    print(f'starting train with {cfg.train.num_epoch} epoch, {cfg._data.paths.num_train_samples} train images, {cfg.train.number_of_train_stages} stages')
    start_epoch = 0
    start_stage = 22
    for epoch in range(cfg.train.num_epoch):
        if epoch < start_epoch:
            continue
        print(f"starting {epoch} epoch")
        for stage in range(cfg.train.number_of_train_stages):
            if stage < start_stage:
                continue
            print(f"Epoch {epoch}: starting {stage + 1} stage of {cfg.train.number_of_train_stages} stages")
            subprocess.run(['python', 'train_rubert.py', f'++stage={stage}'])



if __name__ == "__main__":
    main()

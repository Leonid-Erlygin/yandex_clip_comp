# hydra imports
import hydra
from omegaconf import DictConfig
import argparce

# i2t imports
from i2t.train_rubert import train



@hydra.main(config_path="config", config_name="rubert")
def main(cfg : DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    

    main()

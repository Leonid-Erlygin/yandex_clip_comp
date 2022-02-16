# typing imports
from omegaconf import DictConfig


import sys

# generic imports
import logging
import os
from tqdm import tqdm
import jsonlines

# torch imports
import pytorch_lightning as pl

# custom imports
from i2t.system import I2T
from i2t.data_rubert import get_dataloaders
from i2t.utils import instantiate


# logger = logging.getLogger(__name__)

logger = logging.getLogger("transformers")
logger.setLevel(logging.ERROR)


def train(cfg: DictConfig) -> None:

    # load all needed images for this stage
    i = 0
    train_stage = int(cfg.stage)
    data_stage = []
    assert cfg._data.paths.num_train_samples % cfg.train.number_of_train_stages == 0
    train_size = cfg._data.paths.num_train_samples // cfg.train.number_of_train_stages

    with jsonlines.open(cfg._data.paths.metadata_file) as reader:
        reader = tqdm(reader)
        for obj in reader:
            if i < train_stage * train_size:
                i += 1
                continue
            if i >= (train_stage + 1) * train_size:
                break
            # train_stage * train_size : (train_stage + 1) * train_size
            data_stage.append((obj["image"], obj["queries"]))
            i += 1

    #assert len(data_stage) == train_size

    resume_from_checkpoint = cfg.train.resume_from_checkpoint
    if (resume_from_checkpoint is not None) and (
        not os.path.exists(resume_from_checkpoint)
    ):

        print(
            f"Not using missing checkpoint {resume_from_checkpoint}, starting from scratch..."
        )

        resume_from_checkpoint = None

    callbacks = [instantiate(x) for x in cfg.train.callbacks.values()]
    plugins = [instantiate(x) for x in cfg.train.plugins.values()]

    if resume_from_checkpoint is None:
        load_ckpt = False
    else:
        load_ckpt = True

    # one stage in one epoch
    trainer = pl.Trainer(
        **cfg.train.trainer_params,
        plugins=plugins,
        weights_summary=None,
        callbacks=callbacks,
        logger=instantiate(cfg.train.logger),
        profiler=None,
    )
    
    model = I2T(config=cfg)
    if load_ckpt:
        # load model from checkpoint
        model = model.load_from_checkpoint(cfg.train.resume_from_checkpoint)
    train_dataloader = get_dataloaders(
        data_stage,
        cfg._data.paths.images_directory,
        '/home/devel/mlcup_cv/baseline/text_models/rubert-tiny',
        batch_size=cfg.train.batch_size,
        dataloader_workers=8,
    )
    trainer.fit(train_dataloaders=train_dataloader, model=model)
    trainer.save_checkpoint(cfg.train.resume_from_checkpoint)

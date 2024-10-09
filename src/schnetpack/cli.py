import logging
import os
import uuid
import tempfile
import socket
from typing import List
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.logger import Logger

import schnetpack as spk
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config
from schnetpack.data import BaseAtomsData, AtomsLoader
from schnetpack.train import PredictionWriter
from schnetpack import properties

import time

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

header = """
GULSUM2
   _____      __    _   __     __  ____             __
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host GULSUM!: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            f"Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            f"""
        Config incomplete HELLO! You have to specify at least `data` and `model`!
        For an example, try one of our pre-defined experiments:
        > spktrain data_dir=/data/will/be/here +experiment=qm9
        """
        )
        return

    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"

        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    if config.get("print_config"):
        print_config(config, resolve=False)

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info(f"Seed randomly...")
        seed_everything(workers=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.data,
        train_sampler_cls=str2class(config.data.train_sampler_cls) if config.data.train_sampler_cls else None,
    )
    # datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_10.npz'
    # # datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_20.npz'
    # # datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_40.npz'
    # # datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_60.npz'
    # # datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_80.npz'
    # s_file = np.load(datamodule.split_file)
    # datamodule.num_train = len(s_file['train_idx'])
    # datamodule.num_test = len(s_file['test_idx'])
    # datamodule.num_val = len(s_file['val_idx'])

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    # model = hydra.utils.instantiate(config.model) #IGNORE
    model = torch.load("/home/gulsum/Documents/Surrogates/schnetpack/src/runs/Train100/best_model")

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial"
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
        # limit_predict_batches=0.1,
        # max_epochs=2
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

########################################################################################PREV EXPS#########################################################################################
    # #Change split files
    train_perc = 100
    datamodule.split_file = f'/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_{train_perc}_perc_of_nitrogen_100_perc_of_remaining_test_100_perc_of_nitrogen_100_perc_of_remaining.npz'

    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Train the model
    log.info(f"Starting training for {train_perc}% nitrogen.")
    # trainer.fit(model=task, datamodule=datamodule) #FOR CHECKPOINT COMMENT THIS OUT

    #100% N
    # Evaluate model on test set after training
    log.info("Starting testing for 100% nitrogen.")
    start = time.time()
    trainer.test(model=task, datamodule=datamodule, )
    end = time.time()
    log.info("Testing for 100% nitrogen took " + str(end - start) + " seconds")

    #Change split files 80% N
    datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_100_perc_of_nitrogen_100_perc_of_remaining_test_80_perc_of_nitrogen_100_perc_of_remaining.npz'
    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Evaluate model on test set after training
    log.info("Starting testing for 80% nitrogen.")
    start = time.time()
    trainer.test(model=task, datamodule=datamodule)
    end = time.time()
    log.info("Testing for 80% nitrogen took " + str(end - start) + " seconds")

    #Change split files 60% N
    datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_100_perc_of_nitrogen_100_perc_of_remaining_test_60_perc_of_nitrogen_100_perc_of_remaining.npz'
    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Evaluate model on test set after training
    log.info("Starting testing for 60% nitrogen.")
    start = time.time()
    trainer.test(model=task, datamodule=datamodule)
    end = time.time()
    log.info("Testing for 60% nitrogen took " + str(end - start) + " seconds")

    #Change split files 40% N
    datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_100_perc_of_nitrogen_100_perc_of_remaining_test_40_perc_of_nitrogen_100_perc_of_remaining.npz'
    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Evaluate model on test set after training
    log.info("Starting testing for 40% nitrogen.")
    start = time.time()
    trainer.test(model=task, datamodule=datamodule)
    end = time.time()
    log.info("Testing for 40% nitrogen took " + str(end - start) + " seconds")


    #Change split files 20% N
    datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_100_perc_of_nitrogen_100_perc_of_remaining_test_20_perc_of_nitrogen_100_perc_of_remaining.npz'
    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Evaluate model on test set after training
    log.info("Starting testing for 20% nitrogen.")
    start = time.time()
    trainer.test(model=task, datamodule=datamodule)
    end = time.time()
    log.info("Testing for 20% nitrogen took " + str(end - start) + " seconds")


    #Change split files 0% N
    datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_100_perc_of_nitrogen_100_perc_of_remaining_test_0_perc_of_nitrogen_100_perc_of_remaining.npz'
    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Evaluate model on test set after training
    log.info("Starting testing for 0% nitrogen.")
    start = time.time()
    trainer.test(model=task, datamodule=datamodule)
    end = time.time()
    log.info("Testing for 0% nitrogen took " + str(end - start) + " seconds")

#########################################################100########################################################

    datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_files/split_train_100_perc_of_nitrogen_100_perc_of_remaining_test_100_perc_of_nitrogen_0_perc_of_remaining.npz'
    s_file = np.load(datamodule.split_file)
    datamodule.num_train = len(s_file['train_idx'])
    datamodule.num_test = len(s_file['test_idx'])
    datamodule.num_val = len(s_file['val_idx'])

    # Evaluate model on test set after training
    log.info(f"Starting testing for only nitrogen, Train {train_perc}% N." )
    start = time.time()
    trainer.test(model=task, datamodule=datamodule)
    end = time.time()
    log.info("Testing for only nitrogen took " + str(end - start) + " seconds")

    # datamodule.split_file = '/home/gulsum/Documents/Surrogates/schnetpack/src/schnetpack/splits/split_test_only_nitrogens_included.npz'
    # s_file = np.load(datamodule.split_file)
    # datamodule.num_train = len(s_file['train_idx'])
    # datamodule.num_test = len(s_file['test_idx'])
    # datamodule.num_val = len(s_file['val_idx'])

    # # Evaluate model on test set after training
    # log.info("Starting testing for only nitrogen, Train 0% N.")
    # trainer.test(model=task, datamodule=datamodule)

    # # Evaluate model on test set after training
    # log.info("Starting testing for only nitrogen, Train 20% N.")
    # trainer.test(model=task, datamodule=datamodule)

    # # Evaluate model on test set after training
    # log.info("Starting testing for only nitrogen, Train 40% N.")
    # trainer.test(model=task, datamodule=datamodule)

    # # Evaluate model on test set after training
    # log.info("Starting testing for only nitrogen, Train 60% N.")
    # trainer.test(model=task, datamodule=datamodule)

    # # Evaluate model on test set after training
    # log.info("Starting testing for only nitrogen, Train 80% N.")
    # trainer.test(model=task, datamodule=datamodule)

    # # Evaluate model on test set after training
    # log.info("Starting testing for only nitrogen, Train 100% N.")
    # trainer.test(model=task, datamodule=datamodule)


##########################################################################################################################################################################################
    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info(f"Store best model")
    best_task = type(task).load_from_checkpoint(best_path)
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")


@hydra.main(config_path="configs", config_name="predict", version_base="1.2")
def predict(config: DictConfig):
    log.info(f"Load data from `{config.data.datapath}`")
    dataset: BaseAtomsData = hydra.utils.instantiate(config.data)
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)

    model = torch.load("best_model")

    class WrapperLM(LightningModule):
        def __init__(self, model, enable_grad=config.enable_grad):
            super().__init__()
            self.model = model
            self.enable_grad = enable_grad

        def forward(self, x):
            return model(x)

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            torch.set_grad_enabled(self.enable_grad)
            results = self(batch)
            results[properties.idx_m] = batch[properties.idx][batch[properties.idx_m]]
            results = {k: v.detach().cpu() for k, v in results.items()}
            return results


    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.outputdir, write_interval=config.write_interval, write_idx=config.write_idx_m
            )
        ],
        default_root_dir=".",
        _convert_="partial",
    )
    trainer.predict(
        WrapperLM(model, config.enable_grad),
        dataloaders=loader,
        ckpt_path=config.ckpt_path,
    )

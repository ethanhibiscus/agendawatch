import os
import logging
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from argparse import ArgumentParser
from utils.pytorch_lightning_utils.pytorch_lightning_utils import load_params_from_checkpoint
from utils.pytorch_lightning_utils.callbacks import RunValidationOnStart
from utils import switch_functions
from utils.argparse_init import default_arg_parser, init_parse_argparse_default_params

logging.basicConfig(level=logging.INFO)

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

def main():
    """Initialize all the parsers, before training init."""
    parser = default_arg_parser()
    parser = add_trainer_args(parser)  # Add Trainer args manually
    parser = default_arg_parser(description="docBert", parents=[parser])

    eager_flags = init_parse_argparse_default_params(parser)
    model_class_pointer = switch_functions.model_class_pointer(eager_flags["task_name"], eager_flags["architecture"])
    parser = model_class_pointer.add_model_specific_args(parser, eager_flags["task_name"], eager_flags["dataset_name"])

    hyperparams = parser.parse_args()
    main_train(model_class_pointer, hyperparams, parser)

def add_trainer_args(parser):
    parser.add_argument("--accelerator", type=str, default="gpu", help="Type of accelerator to use, e.g. 'cpu', 'gpu'.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use.")
    parser.add_argument("--gradient_clip_val", type=float, default=0.0, help="Value for gradient clipping.")
    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Percentage of training data to use.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Percentage of validation data to use.")
    parser.add_argument("--limit_test_batches", type=float, default=1.0, help="Percentage of test data to use.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Check validation every n epochs.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients every n batches.")
    return parser

def main_train(model_class_pointer, hparams, parser):
    """Initialize the model, call training loop."""
    seed_everything(hparams.seed)

    if hparams.resume_from_checkpoint not in [None, '']:
        hparams = load_params_from_checkpoint(hparams, parser)

    model = model_class_pointer(hparams)

    logger = TensorBoardLogger(save_dir=model.hparams.hparams_dir, name='', default_hp_metric=False)
    logger.log_hyperparams(model.hparams, metrics={model.hparams.metric_to_track: 0})
    print(f"\nLog directory:\n{model.hparams.hparams_dir}\n")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        mode="min" if "acc" not in hparams.metric_to_track else "max",
        monitor=hparams.metric_to_track,
        dirpath=os.path.join(model.hparams.hparams_dir),
        filename="{epoch}",
        verbose=True,
    )

    trainer = Trainer(
        num_sanity_val_steps=2,
        gradient_clip_val=hparams.gradient_clip_val,
        callbacks=[RunValidationOnStart(), PrintCallback(), checkpoint_callback],
        logger=logger,
        max_epochs=hparams.max_epochs,
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        limit_val_batches=hparams.limit_val_batches,
        limit_train_batches=hparams.limit_train_batches,
        limit_test_batches=hparams.limit_test_batches,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        profiler=SimpleProfiler(),
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
    )

    if not hparams.test_only:
        trainer.fit(model, ckpt_path=hparams.resume_from_checkpoint)
    else:
        if hparams.resume_from_checkpoint is not None:
            model = model.load_from_checkpoint(hparams.resume_from_checkpoint, hparams=hparams, map_location=torch.device("cpu"))
        trainer.test(model)

if __name__ == "__main__":
    main()
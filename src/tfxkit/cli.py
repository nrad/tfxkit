import argparse
import hydra
from omegaconf import DictConfig
from tfxkit.core.logger import setup_logging
from tfxkit.core.model_factory import ModelFactory

# import IPython
import sys

DEFAULT_CONFIG_NAME = "quickstart"
DEFAULT_CONFIG_PATH = "configs"


def get_args():
    parser = argparse.ArgumentParser(description="TFxKit CLI")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive IPython shell",
    )
    args, unknown = parser.parse_known_args()
    return args


def prep_model_factory(cfg):
    mf = ModelFactory(config=cfg)
    mf.builder.compile()
    return mf


def download_example_data():
    from tfxkit.common import tf_utils

    tf_utils.download_example_data()


def run_train(mf, cfg):
    mf.history = mf.trainer.fit(
        validation_split=cfg.training.validation_split,
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        sample_weight=mf.data.sample_weight_train,
    )
    # mf.evaluator.add_test_train_preds()
    return mf


def run_eval(mf, cfg):
    mf.evaluator.add_test_train_preds()
    mf.plotter.run_sequence()
    return mf


def run_train_and_eval(mf, cfg):
    mf = run_train(mf, cfg)
    mf = run_eval(mf, cfg)
    return mf


def run_hyper_tuning(mf, cfg):
    mf = prep_model_factory(cfg)
    outputs = mf.hyper_tuner.run_sequence()
    return mf


@hydra.main(
    config_path=DEFAULT_CONFIG_PATH, config_name=DEFAULT_CONFIG_NAME, version_base=None
)
def main(cfg: DictConfig) -> None:
    print("🚀 TFxKit training...")
    setup_logging(level=cfg.logging.level)

    mf = prep_model_factory(cfg)
    task = cfg.get("task", "train")
    if task == "run":
        run_train_and_eval(mf, cfg)
    elif task == "train":
        run_train_and_eval(mf, cfg)
    elif task == "tune":
        run_hyper_tuning(mf, cfg)
    elif task == "download_example_data":
        download_example_data()
    else:
        raise ValueError(f"Unknown task: {task}")


@hydra.main(
    config_path=DEFAULT_CONFIG_PATH, config_name=DEFAULT_CONFIG_NAME, version_base=None
)
def train_entry(cfg: DictConfig) -> None:
    print("🚀 TFxKit training...")
    setup_logging(level=cfg.logging.level)
    mf = prep_model_factory(cfg)
    run_train_and_eval(mf, cfg)


@hydra.main(
    config_path=DEFAULT_CONFIG_PATH, config_name=DEFAULT_CONFIG_NAME, version_base=None
)
def tune_entry(cfg: DictConfig) -> None:
    print("🚀 TFxKit hyperparameter tuning...")
    setup_logging(level=cfg.logging.level)
    mf = prep_model_factory(cfg)
    run_hyper_tuning(mf, cfg)


if __name__ == "__main__":
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    main()

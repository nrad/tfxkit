import logging
from tfxkit.core.logger import setup_logging
from tfxkit.core.config_loader import ConfigLoader
from tfxkit.core.data_manager import DataManager
from tfxkit.core.model_builder import ModelBuilder
from tfxkit.core.trainer import Trainer
from tfxkit.core.evaluator import Evaluator

logger = logging.getLogger(__name__)


class ModelFactory:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.config
        self.data = DataManager(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.model = self.model_builder
        # self.model = self.model_builder.build
        self.trainer = Trainer(self.config, self.model, self.data)
        self.evaluator = Evaluator(self.config, self.model, self.data)

if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    logger.info("ModelFactory starting up...")
    mf = ModelFactory()
    mf.config_loader.print_config()
    # mf.data_manager.load_df()
    # mf.data_manager.df_trainm

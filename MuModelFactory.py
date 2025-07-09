from ModelFactory import ModelFactory
import utils.muon_embedding as muemb
import utils
from utils import tf_utils
from utils import chill_argparser


class MuModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xy_maker = muemb.xy_maker_muon_embedding2
        self.model_definer = muemb.define_muemb_model


class MuPrimaryModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xy_maker = muemb.xy_maker_muonevent_embedding
        self.model_definer = muemb.define_muevtemb_model


Factories = {
    "muemb": MuModelFactory,
    "muprimaryemb": MuPrimaryModelFactory,
}

if __name__ == "__main__":
    chill_argparser = chill_argparser.ChillArgumentParser()
    chill_argparser.add_argument(
        "--config", type=str, required=True, help="path to config file"
    )
    chill_argparser.add_argument(
        "--factory",
        "-f",
        type=str,
        help="which model factory to use",
        default="muemb",
        choices=list(Factories.keys()),
    )

    import sys

    print("argv:", sys.argv)
    print("----------------")
    args, unknown_args = chill_argparser.parse_arguments()

    print("Known Arguments:", args)
    print("Unknown Arguments:", unknown_args)

    config_path = args.config
    factory = Factories[args.factory]
    mf = factory.load_config(config_path, **unknown_args)

    mf.define_model()
    mf.compile_model()

    mf.model.summary()
    mf.model.event_branch.summary()
    mf.model.muon_branch.summary()

    mf.fit()
    mf.plot_history()
    mf.save_model()
    mf.get_pred(nsample=0)
    mf.plot_predictions(
        weights="sel_flux_weights",
        reweight_train_to_test=False,
        plot_name="predictions",
        nsample=0,
    )
    mf.plot_speed_up(save_res=True)
    mf.plot_speed_up(save_res=False, plot_name="speed_up_train", df=mf.df_train_sample)
    # mf.plot_spe
    mf.generate_html()

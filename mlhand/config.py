from sklearn.linear_model import LogisticRegression
from frozendict import frozendict
from skopt.space import Real

# prepared in ./notebooks/data-prep-extract-all-eids-with-imaging-data
DATA_EIDS_WITH_IMAGING_DATA = "./data/ukb/eids_with_complete_imaging_data.csv"

# prepared by Xiangzhen ...
DATA_IMAGING_DATA = (
    "/data/clusterfs/lag/projects/lg-ukbiobank/primary_data/imaging_data"
)

# prepared in ./notebooks/data-prep-handedness-structure-diffusion.ipynb
DATA_PREPROCESSED_STRCTURE_DIFFUSION = (
    "./data/ukb/processed-handedness-structure-diffusion.csv"
)

# prepared in ./notebooks/data-prep-controlled-variables
# noting that in that notebook, these variables are named `confound`.
DATA_CONTROLLED_VARIABLES = "./data/ukb/controlled-variables.csv"

# this file is from https://www.fmrib.ox.ac.uk/ukbiobank/group_means/edge_list_d100.txt.
# it is stored locally for conveneice; md5sum is 35375ff285f33ce7e52e014eef198894
RESOURCE_CORRELATION_EDGE_LIST = "./resources/ukb-edge-list-d100.txt"

# this is the number of participants with rfmri data
TOTAL_PARTICIPANTS = 36024

# these numbers are arbinary but we keep them here for transparency
RANDOM_SEEDS = [23, 5, 107]

NUM_FEATURES = frozendict(
    dict(
        # actuall controlled variables is 20
        # but we convert imaging center using one hot, so we have 20 - 1 + 3 columns
        # where 3 is the number of centers.
        controlled_variables=22,
        structure=198,
        diffusion=432,
        rfmri_component_amplitudes_25=21,
        rfmri_full_correlation_25=210,
        rfmri_partial_correlation_25=210,
        rfmri_component_amplitudes_100=55,
        rfmri_full_correlation_100=1485,
        rfmri_partial_correlation_100=1485,
    )
)

TRAINING_PARMS = frozendict(
    dict(
        OUTER_FOLDS=10,
        INNER_FOLDS=5,
        HYPER_OPT_RUN=5,
    )
)

MODEL_TYPE_CONFIG = frozendict(
    dict(
        logistic=dict(
            cls=LogisticRegression,
            parameters={
                "class_weight": "balanced",
                "max_iter": 1000,
                "tol": 0.001,
                "penalty": "l2",
                "verbose": 0,
            },
            hyperparams={"model__C": Real(1e-6, 1e6, prior="log-uniform")},
        )
    )
)
 

from os import path
import re

import pandas as pd
import numpy as np

from mlhand import config, utils

from tqdm import tqdm

DATA_PROVIDER = dict()

RX_FMRI_IC_PARTIAL_CORR = re.compile("rfmri_(?:not_)?ic(\d+)_partial_correlation_100")


def number_features_for_modality(modality):
    if modality in config.NUM_FEATURES:
        return config.NUM_FEATURES[modality]
    elif re.match(RX_FMRI_IC_PARTIAL_CORR, modality):
        if "_not_" in modality:
            return config.NUM_FEATURES["rfmri_partial_correlation_100"] - 54
        else:
            return 54
    else:
        raise ValueError(f"We don't have number of features for {modality}.")


def register_modality(name):
    """Decorator to register a data modality provider."""

    def wrapped(func):
        """Wrapped function to register a data modality provider with name `name`"""
        DATA_PROVIDER[name] = func
        return func

    return wrapped


def get_unimodal_data(modality):
    if "rfmri_ic" in modality:
        # this is dataset is based on rfmri-partial-correlation-100.
        # it is created on the fly.
        ic_id = re.match(RX_FMRI_IC_PARTIAL_CORR, modality).group(1)
        ic_id = int(ic_id)
        df = get_rfmri_ic_partial_correlation_features(ic_id)
    elif "rfmri_not_ic" in modality:
        # this is dataset is based on rfmri-partial-correlation-100.
        # it is created on the fly.
        ic_id = re.match(RX_FMRI_IC_PARTIAL_CORR, modality).group(1)
        ic_id = int(ic_id)
        df = get_rfmri_feature_not_include_ic_partial_correlation_features(ic_id)
    else:
        df = DATA_PROVIDER[modality]()

    np.testing.assert_equal(df.shape[0], config.TOTAL_PARTICIPANTS, verbose=True)

    return df


def get_multimodal_data(modalities):
    df_base = get_unimodal_data(modalities[0])

    np.testing.assert_equal(df_base.shape[0], config.TOTAL_PARTICIPANTS, verbose=True)

    for modality in modalities[1:]:
        df_modality = get_unimodal_data(modality)
        df_modality = df_modality.drop("HandednessBin", axis=1)

        df_base = df_base.merge(df_modality, on="eid")

    np.testing.assert_equal(df_base.shape[0], config.TOTAL_PARTICIPANTS, verbose=True)
    np.testing.assert_equal(
        df_base.shape[1],
        # plus 2 for `eid` and `HandednessBin`
        2 + np.sum(list(map(lambda m: number_features_for_modality(m), modalities))),
        verbose=True,
    )

    return df_base


def get_covariates(columns):
    """Filter out target and eid columns

    Args:
        columns (list)

    Returns:
        [list]: covariates
    """

    return list(
        filter(lambda x: x not in ["eid", "Handedness", "HandednessBin"], columns)
    )


def get_structure_columns():
    df = load_raw_data_with_binary_handedness()
    structure_cols = list(
        filter(
            lambda x: np.sum(list(c in x for c in ["Area", "thickness", "Volume"])),
            df.columns,
        )
    )
    np.testing.assert_equal(
        len(structure_cols), config.NUM_FEATURES["structure"], verbose=True
    )

    return structure_cols


def get_diffusion_columns():
    df = load_raw_data_with_binary_handedness()

    all_covariate_columns = get_covariates(df.columns)

    structure_cols = get_structure_columns()

    return list(set(all_covariate_columns).difference(structure_cols))


def load_raw_data():
    df = pd.read_csv(config.DATA_PREPROCESSED_STRCTURE_DIFFUSION)

    df_eids_with_imaging_data = pd.read_csv(config.DATA_EIDS_WITH_IMAGING_DATA)

    df = df.merge(df_eids_with_imaging_data, on="eid")

    np.testing.assert_equal(
        df.shape[0],
        config.TOTAL_PARTICIPANTS,
        err_msg="Number of Participants with imaging data is not correct",
        verbose=True,
    )

    # plus 2 for `eid`, `handedness`
    np.testing.assert_equal(
        df.shape[1],
        config.NUM_FEATURES["structure"] + config.NUM_FEATURES["diffusion"] + 2,
        verbose=True,
    )

    return df


def load_raw_data_with_binary_handedness():
    df = load_raw_data()

    # From UKB, Handedness == 1 means Right-Handed (RH)
    # See: https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=100430
    # We set RH (the majority group) to have HandednessBin=0,
    # and the LH and MH groups to have =1.
    df["HandednessBin"] = np.where(df.Handedness == 1, 0, 1)

    df = df.drop("Handedness", axis=1)

    assert set(df.columns) == set(get_covariates(df) + ["eid", "HandednessBin"])
    np.testing.assert_equal(
        df.shape[0],
        config.TOTAL_PARTICIPANTS,
        err_msg="Number of Participants is not correct",
        verbose=True,
    )

    np.testing.assert_equal(
        len(get_covariates(df.columns)),
        config.NUM_FEATURES["structure"] + config.NUM_FEATURES["diffusion"],
        verbose=True,
    )

    return df


def get_raw_data_with_eid_and_binary_handedness():
    df = load_raw_data_with_binary_handedness()

    return df[["eid", "HandednessBin"]]


@register_modality("structure")
def modality_structure():
    df = load_raw_data_with_binary_handedness()

    df = df[["eid", "HandednessBin"] + get_structure_columns()]

    np.testing.assert_equal(
        len(df.columns), 2 + config.NUM_FEATURES["structure"], verbose=True
    )

    return df


@register_modality("diffusion")
def modality_diffusion():
    df = load_raw_data_with_binary_handedness()

    df = df[["eid", "HandednessBin"] + get_diffusion_columns()]

    np.testing.assert_equal(
        len(df.columns), 2 + config.NUM_FEATURES["diffusion"], verbose=True
    )

    return df


@register_modality("controlled_variables")
def modality_controlled_variables():
    df_raw = get_raw_data_with_eid_and_binary_handedness()

    df = pd.read_csv(config.DATA_CONTROLLED_VARIABLES)

    df = df_raw.merge(df, on="eid")

    np.testing.assert_equal(df.shape[0], config.TOTAL_PARTICIPANTS, verbose=True)

    # plus two here is for eid and HandednessBin.
    np.testing.assert_equal(
        df.shape[1], 2 + config.NUM_FEATURES["controlled_variables"], verbose=True
    )

    return df


def load_rfmri_data(field_name, field_id):
    df_eid_handedness = get_raw_data_with_eid_and_binary_handedness()

    df_imaging = read_dataset_imaging_data(df_eid_handedness.eid.values, field_id)

    df = df_eid_handedness.merge(df_imaging, on="eid")

    np.testing.assert_equal(df.shape[0], config.TOTAL_PARTICIPANTS, verbose=True)

    np.testing.assert_equal(
        df.shape[1],
        config.NUM_FEATURES[field_name] + 2,
        verbose=True,
    )

    return df


@register_modality("rfmri_component_amplitudes_25")
def modality_rfmri_component_amplitudes_25():
    return load_rfmri_data("rfmri_component_amplitudes_25", "25754_2_0")


@register_modality("rfmri_full_correlation_25")
def modality_rfmri_full_correlation_25():
    return load_rfmri_data("rfmri_full_correlation_25", "25750_2_0")


@register_modality("rfmri_partial_correlation_25")
def modality_rfmri_partial_correlation_25():
    return load_rfmri_data("rfmri_partial_correlation_25", "25752_2_0")


@register_modality("rfmri_component_amplitudes_100")
def modality_rfmri_component_amplitudes_100():
    return load_rfmri_data("rfmri_component_amplitudes_100", "25755_2_0")


@register_modality("rfmri_full_correlation_100")
def modality_rfmri_full_correlation_100():
    return load_rfmri_data("rfmri_full_correlation_100", "25751_2_0")


@register_modality("rfmri_partial_correlation_100")
def modality_rfmri_partial_correlation_100():
    return load_rfmri_data("rfmri_partial_correlation_100", "25753_2_0")


def read_individual_imaging_data(eid, field_id, verbose=False):

    fp = f"{config.DATA_IMAGING_DATA}/{eid}/{eid}_{field_id}.txt"

    if not path.exists(fp):
        raise ValueError(f"{eid} doesn't have imaging data for field_id={field_id}")

    with open(fp) as fh:
        line = fh.readline().strip()
        st = re.split(r"\s+", line)
        if not st:
            if verbose:
                print(eid, st)

            raise ValueError(f"{eid}'s field_id={field_id} is empty.")

        data = np.array(st).astype(np.float)

    return data


def read_dataset_imaging_data(eids, field_id="25755_2_0", verbose=False):

    individual_rows = []

    for eid in tqdm(eids, desc=f"Loading {field_id}"):
        components = read_individual_imaging_data(
            eid, field_id=field_id, verbose=verbose
        )

        individual_rows.append(dict(eid=eid, components=components))

    field_id = field_id.split("_")[0]
    labels = list(
        map(
            lambda i: f"F{field_id}_E{i}",
            range(1, individual_rows[0]["components"].shape[0] + 1),
        )
    )

    data = []
    for row in individual_rows:
        data.append(dict(eid=row["eid"], **dict(zip(labels, row["components"]))))
    df = pd.DataFrame(data)

    return df


def get_rfmri_ic_partial_correlation_features(ic_id):
    df_pc = modality_rfmri_partial_correlation_100()

    relevant_columns = utils.get_ic_related_features(ic_id, "25753")

    df_ic = df_pc[["eid", "HandednessBin"] + relevant_columns]

    np.testing.assert_equal(df_ic.shape[1], 2 + 54, verbose=1)

    return df_ic


def get_rfmri_feature_not_include_ic_partial_correlation_features(ic_id):
    df_pc = modality_rfmri_partial_correlation_100()

    relevant_columns = utils.get_ic_related_features(ic_id, "25753")
    cols = set(df_pc.columns.values).difference(relevant_columns)

    df_ic = df_pc[cols]

    df = pd.read_csv(config.RESOURCE_CORRELATION_EDGE_LIST, sep=" ")
    feature_ids = df[(df.Node1 != ic_id) & (df.Node2 != ic_id)].EdgeNumber.values

    np.testing.assert_equal(
        sorted(list(map(lambda x: f"F25753_E{x}", feature_ids))),
        sorted(list(set(df_ic.columns).difference(["eid", "HandednessBin"]))),
        verbose=1,
    )

    np.testing.assert_equal(
        df_ic.shape[1],
        2 + config.NUM_FEATURES["rfmri_partial_correlation_100"] - 54,
        verbose=1,
    )

    return df_ic

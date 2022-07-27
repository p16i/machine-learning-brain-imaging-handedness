from os import sep
import subprocess
import pandas as pd

from mlhand import config

# taken from https://stackoverflow.com/a/21901260
def get_git_revision_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_ic_related_features(ic_id, feature_prefix):
    # ic_id is from 1 to 55.
    # feature_prefix is {FC-100=25751, PC-100=25753}

    assert 1 <= ic_id <= 55, f"IC-{ic_id} is not valid (the valid range is [1, 55]"

    assert feature_prefix in [
        "25751",
        "25753",
    ], "we only decode features from UKB Category [25751, 25753]"

    df = pd.read_csv(config.RESOURCE_CORRELATION_EDGE_LIST, sep=" ")

    feature_ids = df[(df.Node1 == ic_id) | (df.Node2 == ic_id)].EdgeNumber.values

    # we hardcode here because we are only interested in rfmri-*-correlation from 100 ICs
    # and UKB calculates them only from good ICs (55 ICs).
    assert len(feature_ids) == 54

    feature_cols = list(map(lambda i: f"F{feature_prefix}_E{i}", feature_ids))

    return feature_cols

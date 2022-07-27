import os
import click

import json
import re

import numpy as np

from joblib import dump

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
)
from skopt import BayesSearchCV


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# Because we limit the number of optimization iters,
# Sklearn's optimizer often raises convergence warning.
simplefilter("ignore", category=ConvergenceWarning)

from tqdm import tqdm
from datetime import datetime

from mlhand import config, dataset, utils


def objective_scorer(est, X, y):
    pred_prob = est.predict_proba(X)[:, 1]
    return objective_func(y, pred_prob)


def objective_func(label, score):
    return roc_auc_score(label, score)


@click.command()
@click.option(
    "--artifact-dir", default="./artifacts/tmp", help="where to save artifact"
)
@click.option(
    "--modalities",
    required=True,
    help="Modalities e.g., specifying it with structure,diffusion will uses these modalities to train a model",
)
@click.option(
    "--model-type",
    default="logistic",
    type=click.Choice(["logistic", "svm-linear", "svm-rbf"], case_sensitive=False),
)
@click.option(
    "--random-seed-id",
    default=0,
    help="This is the index of the random seed not the actual value; the value is in mlhand/config.py",
)
@click.option(
    "--quick-run",
    is_flag=True,
    default=False,
    required=False,
    help="Subset the data into only 1000 rows to train. This is for development purproses",
)
def train(modalities, model_type, artifact_dir, quick_run, random_seed_id):
    modalities = modalities.split(",")
    click.echo(f"Training `{model_type}` with `{len(modalities)}` modalities")

    num_features = 0
    for modality in modalities:
        modality_num_features = dataset.number_features_for_modality(modality)
        click.echo(f"  - {modality} ({modality_num_features} features)")
        num_features += modality_num_features

    click.echo(f"Total Features : {num_features} features")

    model_slug = f"{model_type}--{'-'.join(modalities)}"
    artifact_path = f"{artifact_dir}/{model_slug}"
    click.echo(f"Artifact Directory: {artifact_path}")
    os.makedirs(artifact_path, exist_ok=True)

    df_data = dataset.get_multimodal_data(modalities)
    print(f"Total columns (including eid and HandednessBin): {df_data.shape[1]}")

    np.testing.assert_equal(df_data.shape[0], config.TOTAL_PARTICIPANTS, verbose=True)

    if quick_run:
        click.echo("[quick-run=True]: We select only 1000 rows to train.")
        df_data = df_data[:1000]
        np.testing.assert_equal(df_data.shape[0], 1000, verbose=True)

    click.echo(f"Random Seed: {config.RANDOM_SEEDS[random_seed_id]}")

    random_seed = config.RANDOM_SEEDS[random_seed_id]

    outer_spliter = StratifiedKFold(
        n_splits=config.TRAINING_PARMS["OUTER_FOLDS"],
        random_state=random_seed,
        shuffle=True,
    )

    click.echo(
        f"Performing: NestedCV(outer_fold={config.TRAINING_PARMS['OUTER_FOLDS']}, inner_fold={config.TRAINING_PARMS['INNER_FOLDS']}) with {config.TRAINING_PARMS['HYPER_OPT_RUN']} runs"
    )

    covariates_cols = dataset.get_covariates(df_data.columns)

    np.testing.assert_equal(len(covariates_cols), num_features, verbose=True)

    X, y = df_data[covariates_cols], df_data.HandednessBin

    statistics = dict(aucs=[], fprs=[], tprs=[])
    predictions = []  # (eids, pred_probatility, HandednessBin)

    os.makedirs(f"{artifact_path}/best_models", exist_ok=True)

    click.echo(f"Training {model_slug}")
    training_start = datetime.now()
    for outer_loop_ix, (train_index, test_index) in enumerate(
        tqdm(
            outer_spliter.split(X, y),
            total=config.TRAINING_PARMS["OUTER_FOLDS"],
            desc="Outer Loop",
        )
    ):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        model_configure = config.MODEL_TYPE_CONFIG[model_type]

        base_model = Pipeline(
            [
                ("preprocessing", MinMaxScaler()),
                ("model", model_configure["cls"](**model_configure["parameters"])),
            ]
        )

        opt = BayesSearchCV(
            base_model,
            model_configure["hyperparams"],
            n_iter=config.TRAINING_PARMS["HYPER_OPT_RUN"],
            cv=StratifiedKFold(
                n_splits=config.TRAINING_PARMS["INNER_FOLDS"],
                random_state=random_seed,
                shuffle=True,
            ),
            scoring=objective_scorer,
        )

        opt.fit(X_train, y_train)

        best_estimator = opt.best_estimator_

        prediction_probs = best_estimator.predict_proba(X_test)[:, 1]

        test_set_eids = df_data.iloc[test_index]["eid"]

        fpr, tpr, _ = roc_curve(y_test, prediction_probs)
        auc = roc_auc_score(y_test, prediction_probs)

        statistics["aucs"].append(auc.tolist())
        statistics["fprs"].append(fpr.tolist())
        statistics["tprs"].append(tpr.tolist())

        predictions.append(
            dict(
                eids=test_set_eids.tolist(),
                actual_y=y_test.tolist(),
                pred_prob=prediction_probs.tolist(),
            )
        )

        dump(
            best_estimator,
            f"{artifact_path}/best_models/outer-loop-{outer_loop_ix}.joblib",
        )

    training_end = datetime.now()

    time_took = (training_end - training_start).seconds
    click.echo(f"Training took: {time_took}s")

    click.echo(f"Artifacts are saved to {artifact_path}/*")

    with open(f"{artifact_path}/statistics.json", "w") as fh:
        json.dump(statistics, fh, sort_keys=True, indent=4)

    with open(f"{artifact_path}/testset_predictions.json.confidential", "w") as fh:
        json.dump(predictions, fh, sort_keys=True, indent=4)

    with open(f"{artifact_path}/config.json", "w") as fh:
        json.dump(
            dict(
                commit=utils.get_git_revision_hash(),
                random_seed_value=random_seed,
                date=f"{datetime.now()}",
                modalities=",".join(modalities),
                time_took=time_took,
            ),
            fh,
            sort_keys=True,
            indent=4,
        )


if __name__ == "__main__":
    train()

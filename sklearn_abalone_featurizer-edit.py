from __future__ import print_function

import argparse
import csv
import json
import os
import shutil
import sys
import time
from io import StringIO
import pickle as pkl
import subprocess
import numpy as np
import logging
import joblib
import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)


subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

import boto3
# boto3.client('sagemaker', region_name='ap-southeast-2')   

# client = boto3.client('kinesis', config=my_config
import sagemaker

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, OneHotEncoder, StandardScaler

boto_session = boto3.Session()
boto_fs_client = boto_session.client(service_name='sagemaker-featurestore-runtime', 
                                     region_name='ap-southeast-2')

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sex",  # M, F, and I (infant)
    "length",  # Longest shell measurement
    "diameter",  # perpendicular to length
    "height",  # with meat in shell
    "whole_weight",  # whole abalone
    "shucked_weight",  # weight of meat
    "viscera_weight",  # gut weight (after bleeding)
    "shell_weight",
]  # after being dried

label_column = "rings"

feature_columns_dtype = {
    "sex": "category",
    "length": "float64",
    "diameter": "float64",
    "height": "float64",
    "whole_weight": "float64",
    "shucked_weight": "float64",
    "viscera_weight": "float64",
    "shell_weight": "float64",
}

label_column_dtype = {"rings": "float64"}  # +1.5 gives the age in years


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(input_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    raw_data = [
        pd.read_csv(
            file,
            header=None,
            names=feature_columns_names + [label_column],
            dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
        )
        for file in input_files
    ]
    concat_data = pd.concat(raw_data)

    # Labels should not be preprocessed. predict_fn will reinsert the labels after featurizing.
    concat_data.drop(label_column, axis=1, inplace=True)

    # This section is adapted from the scikit-learn example of using preprocessing pipelines:
    #
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    #
    # We will train our classifier with the following features:
    # Numeric Features:
    # - length:  Longest shell measurement
    # - diameter: Diameter perpendicular to length
    # - height:  Height with meat in shell
    # - whole_weight: Weight of whole abalone
    # - shucked_weight: Weight of meat
    # - viscera_weight: Gut weight (after bleeding)
    # - shell_weight: Weight after being dried
    # Categorical Features:
    # - sex: categories encoded as strings {'M', 'F', 'I'} where 'I' is Infant
    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
            ("cat", categorical_transformer, make_column_selector(dtype_include="category")),
        ]
    )

    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")

## here we add another input channel, will input data from sagemaker online features store 
def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        print('Input data: ', df)
        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_columns_names + [label_column]
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names
        elif len(df.columns) < len(feature_columns_names):
                    #params = input_data.split(',')
                    fg_name = df.iloc[0,0]
                    print('fg_name: ', fg_name)
                    input_feat_id = df.iloc[0,1]
                    print(input_feat_id)
                    rec = boto_fs_client.get_record(FeatureGroupName=fg_name, RecordIdentifierValueAsString=str(input_feat_id),FeatureNames=feature_columns_names)
                    feats = rec.get('Record', None)
                    print(feats)
                    features= [','.join(i['ValueAsString'] for i in feats)]
                    df = pd.DataFrame([sub.split(",") for sub in features], index=None)
                    df.columns = feature_columns_names
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a  preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if label_column in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[label_column], axis=1)
    else:
        # Return only the set of features
        return features

def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def train_model(model, X, y):
    '''
    Function to train the model using the input data

    Inputs:
        model: the model that needs to be trained
        X: features to train the model
        y: targets to train the model

    Output:
        a trained model
    '''

    model.fit(X, y)
    return model

def evaluate_model(model, X, y, metric):
    '''
    Function to evaluate the model results with the predicted values of the model and ground truth based on the specified metric

    Inputs:
        model: trained model
        X: features for the model to inference
        y: targets to evaluate the predicted results
        metric: metric to evaluate the difference between ground truth and predicted values
    '''

    y_pred = model.predict(X)
    print(metric(y, y_pred))

    return

def create_sklearn_pipeline(pipeline_object, pipeline_name = 'sklearn_pipeline'):
    '''
    Function to create a sklearn pipeline using a sklearn object

    Inputs:
        pipeline_object: an sklearn object that can be used to create the sklearn pipeline
        pipeline_name: name of the sklearn pipeline

    Output:
        sklearn pipeline object
    '''

    return Pipeline(steps=[(pipeline_name, pipeline_object)])

def get_num_cat_cols(df, skip_cols):
    '''
    Function to get the numerical and categorical columns from a dataframe

    Inputs:
        df: dataframe to get the columns from
        skip_cols: columns that need to be skipped

    Outputs:
        numerical_columns: list consisting of numerical columns
        categorical_columns: list consisting of categorical values
    '''

    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    numerical_columns = [col for col in numerical_columns if col not in skip_cols]
    categorical_columns = [col for col in categorical_columns if col not in skip_cols]

    return numerical_columns, categorical_columns

def build_preprocessing_model_pipeline(X, skip_cols, numerical_scal)
    
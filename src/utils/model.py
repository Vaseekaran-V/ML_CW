import numpy as np
import pandas as pd

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

def create_sklearn_pipeline():
    '''
    Function to create a sklearn pipeline using the sklearn Pipeline class
    '''

    pipeline = None #tbd

    return pipeline
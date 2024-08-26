import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LinearRegression

from utils.data_preprocess import preprocess_data

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

def build_preprocessing_model_pipeline(X, skip_cols, numerical_scaler = None,categorical_scaler = None,
                                       model_sales = None, model_qty = None) -> Pipeline:
    '''
    Build a complete pipeline with preprocessing and model

    Inputs:
        X: the input features with column names
        skip_cols: columns to skip
        numerical_scaler: a scaling object to scale the numerical fields
        categorical_scaler: a scaling object to encode the categorical fields
        ml_model: model object

    Output:
        trained model pipeline
    '''
    # if numerical_scaler == None:
    #     numerical_scaler = RobustScaler()

    # if categorical_scaler == None:
    #     categorical_scaler = OneHotEncoder(drop = 'first', handle_unknown = 'ignore')

    if model_sales == None:
        model_sales = LinearRegression()

    if model_qty == None:
        model_qty = LinearRegression()
    
    numerical_columns, categorical_columns = get_num_cat_cols(X, skip_cols)

    numerical_scaler_pipeline = create_sklearn_pipeline(numerical_scaler, 'numerical_scaler')
    categorical_scaler_pipeline = create_sklearn_pipeline(categorical_scaler, 'categorical_scaler')
    model_sales_pipeline = create_sklearn_pipeline(model_sales, 'model_sales')
    model_qty_pipeline = create_sklearn_pipeline(model_qty, 'model_qty')


    preprocessor = ColumnTransformer(transformers = [
        ('numerical_processor', numerical_scaler_pipeline, numerical_columns),
        ('categorical_processor', categorical_scaler_pipeline, categorical_columns)],
        remainder = 'drop', force_int_remainder_cols = False)
    
    preprocessing_model_sales_pipeline = Pipeline(steps = [('preprocessor', preprocessor), ('model', model_sales_pipeline)])
    preprocessing_model_qty_pipeline = Pipeline(steps = [('preprocessor', preprocessor), ('model', model_qty_pipeline)])

    return preprocessing_model_sales_pipeline, preprocessing_model_qty_pipeline

def create_production_df(start_date, end_date, depts_list, stores_list):
    '''
    Function to create a production dataframe, with date, item depts and stores. To be used for inference

    Inputs:
        start_date: starting date of the dataframe
        end_date: ending date of the dataframe
        depts_list: list of departments
        stores_list: list of stores

    Output:
        dataframe with date, item depts and stores
    '''
    date_range = pd.date_range(start_date, end_date)

    combinations = list(product(date_range, depts_list, stores_list))

    production_df = pd.DataFrame(combinations, columns = ['date_id', 'item_dept', 'store'])

    return production_df

def recursive_forecasting(historical_df, stores_list, depts_list, production_df,
                          model_sales = None, model_qty = None, max_window_length = 8):
    '''
    Function to forecast recursively for the specified production df

    Inputs:
        historical_df: dataframe with historical data
        stores_list: list of stores
        depts_list: list of departments
        production_df: dataframe to be created
        model_sales: model to predict sales
        model_qty: model to predict item quantity
        max_window_length: max window length of historical df to consider when creating time series features
    Output:
        dataframe with forecasted values
    '''
    count = 0
    for current_date in production_df['date_id'].unique():
        for dept in depts_list:
            for store in stores_list:

                relevant_historical_data = historical_df[(historical_df['store'] == store) & (historical_df['item_dept'] == dept)]
                
                current_day_data = production_df[
                    (production_df['date_id'] == current_date) & 
                    (production_df['item_dept'] == dept) & 
                    (production_df['store'] == store)
                ]

                combined_df = pd.concat([relevant_historical_data, current_day_data], ignore_index=True).tail(max_window_length)
                combined_df_processed = preprocess_data(combined_df, num_lags=3, rolling_window_size=3)

                # current_day_data['net_sales'] = model_sales.predict(combined_df_processed.tail(1).drop(columns = ['net_sales', 'item_qty']))
                # current_day_data['item_qty'] = model_qty.predict(combined_df_processed.tail(1).drop(columns = ['net_sales', 'item_qty']))

                #Tester function to see if the function works
                current_day_data['net_sales'] = count
                current_day_data['item_qty'] = count

                count += 1

                production_df.loc[
                    (production_df['date_id'] == current_date) &
                    (production_df['item_dept'] == dept) &
                    (production_df['store'] == store), ['net_sales', 'item_qty']
                ] = current_day_data[['net_sales', 'item_qty']].values

                historical_df = pd.concat([historical_df, current_day_data], ignore_index=True)

    return production_df

def generate_forecasting_df(start_date, end_date, depts_list, stores_list, historical_df,
                            model_sales = None, model_qty = None, max_window_length = 8):
    '''
    Function to create the forecasting df and doing recursive forecasting

    Inputs:
        start_date: starting date of the dataframe
        end_date: ending date of the dataframe
        depts_list: list of departments
        stores_list: list of stores
        historical_df: dataframe with historical data
        model_sales: model to predict sales
        model_qty: model to predict item quantity
        max_window_length: max window length of historical df to consider when creating time series features

    Output:
        dataframe with forecasted values
    '''
    production_df = create_production_df(start_date, end_date, depts_list, stores_list)

    forecasted_df = recursive_forecasting(historical_df=historical_df, stores_list=stores_list,
                                          depts_list=depts_list, production_df=production_df,
                                          model_sales=model_sales, model_qty=model_qty,
                                          max_window_length=max_window_length)
    return forecasted_df

def create_train_test_set(df, date_col = 'date_id', test_date_start = '2022-02-01'):
    '''
    Function to create train and test data using date range
    
    Inputs:
        df: dataframe to split
        date_col: date column
        test_date_start: the starting date of the test set (the dates before will be training, and after will be testing)

    Outputs:
        train and test dataframes
    '''
    df_process = df.copy()

    df_train = df_process[df_process[date_col] < test_date_start]
    df_test = df_process[df_process[date_col] >= test_date_start]

    return df_train, df_test

def create_X_y(df, y, drop_cols = ['date_id', 'item_dept', 'store', 'net_sales', 'item_qty']):
    '''
    Function to create X and y from the given dataframe

    Inputs:
        df: dataframe
        y: target feature
        drop_cols: cols to drop (other than y and training features)

    Outputs:
        training features and target (y)
    '''

    df_process = df.copy()

    target = df_process[y]
    features = df_process.drop(columns = drop_cols)

    return features, target

def create_X_and_targets_sales_qty(df, drop_cols = ['date_id', 'item_dept', 'store', 'net_sales', 'item_qty'],
                                   net_sales_col = 'net_sales', item_qty_col = 'item_qty'):
    '''
    Function to create training features and the targets to forecast sales and item quantity

    Inputs:
        df: dataframe
        drop_cols: cols to drop (other than y and training features)
        net_sales_col: name of net sales column
        item_qty_col: name of item quantities sold column

    Outputs:
        training features and targets (net sales and item quantities sold)

    '''

    df_process = df.copy()

    features, y_net_sales = create_X_y(df = df_process, y = net_sales_col, drop_cols = drop_cols)
    _, y_item_qty = create_X_y(df = df_process, y = item_qty_col, drop_cols = drop_cols)

    return features, y_net_sales, y_item_qty

def create_training_testing(df, date_col = 'date_id', test_date_start = '2022-02-01',
                            drop_cols = ['date_id', 'item_dept', 'store', 'net_sales', 'item_qty'], net_sales_col = 'net_sales',
                            item_qty_col = 'item_qty'):
    '''
    Function to create final train test splits

    Inputs:
        df: dataframe to split
        date_col: date column
        test_date_start: the starting date of the test set (the dates before will be training, and after will be testing)
        drop_cols: cols to drop (other than y and training features)
        net_sales_col: name of net sales column
        item_qty_col: name of item quantities sold column
    '''

    df_process = df.copy()

    df_train, df_test = create_train_test_set(df_process, date_col = date_col, test_date_start = test_date_start)

    train_feat, train_net_sales, train_item_qty = create_X_and_targets_sales_qty(df_train, drop_cols = drop_cols,
                                                                                 net_sales_col = net_sales_col,
                                                                                 item_qty_col = item_qty_col)
    
    test_feat, test_net_sales, test_item_qty = create_X_and_targets_sales_qty(df_test, drop_cols = drop_cols,
                                                                                 net_sales_col = net_sales_col,
                                                                                 item_qty_col = item_qty_col)
    
    train_dict = {'train_features': train_feat, 'train_net_sales': train_net_sales, 'train_item_qty': train_item_qty}
    test_dict = {'train_features': test_feat, 'train_net_sales': test_net_sales, 'train_item_qty': test_item_qty}

    return train_dict, test_dict



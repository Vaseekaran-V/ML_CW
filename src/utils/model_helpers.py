import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

from utils.DataPreprocessPipeline import DataPreprocessPipeline
from utils.SalesItemQtyModel import SalesItemQtyModel
import mlflow
import mlflow.sklearn


def store_level_results(test_df, pred_df, scoring = mean_absolute_percentage_error):
    '''
    evaluating the score for the predictions based on ground truth for store level

    Inputs:
        test_df: ground truth dataframe with stores, departments, and dates, and sales and item qty
        pred_df: predicted dataframe with stores, departments, and dates, and sales and item qty
        scoring: scoring metric
    '''
    test_df_stores = test_df.groupby(['date_id', 'store'])[['net_sales', 'item_qty']].sum().reset_index()
    pred_df_stores = pred_df.groupby(['date_id', 'store'])[['net_sales', 'item_qty']].sum().reset_index()

    sales_mape = scoring(test_df_stores['net_sales'], pred_df_stores['net_sales'])
    item_qty_mape = scoring(test_df_stores['item_qty'], pred_df_stores['item_qty'])

    print(f'Overall MAPE score for Sales: {sales_mape}')
    print(f'Overall MAPE score for Item Qty: {item_qty_mape}\n')

    for store in test_df_stores['store'].unique():
        test_df_store_single = test_df_stores[test_df_stores['store'] == store]
        pred_df_store_single = pred_df_stores[pred_df_stores['store'] == store]

        sales_mape_store = scoring(test_df_store_single['net_sales'], pred_df_store_single['net_sales'])
        item_qty_mape_store = scoring(test_df_store_single['item_qty'], pred_df_store_single['item_qty'])

        print(f'For Store {store}, MAPE for predicting sales each day: {sales_mape_store}')
        print(f'For Store {store}, MAPE for predicting item qty each day: {item_qty_mape_store}\n')

        for item_dept in test_df['item_dept'].unique():
            test_df_store_dept = test_df[(test_df['store'] == store) & (test_df['item_dept'] == item_dept)]
            pred_df_store_dept = pred_df[(pred_df['store'] == store) & (pred_df['item_dept'] == item_dept)]

            sales_mape_store_dept = scoring(test_df_store_dept['net_sales'], pred_df_store_dept['net_sales'])
            item_qty_mape_store_dept = scoring(test_df_store_dept['item_qty'], pred_df_store_dept['item_qty'])

            print(f'For Store {store} and Department {item_dept}, MAPE for predicting sales each day: {sales_mape_store_dept}')
            print(f'For Store {store} and Department {item_dept}, MAPE for predicting item qty each day: {item_qty_mape_store_dept}\n')



def get_results_fitted(train_dict, valid_dict, sales_item_qty_model):
    '''
    Function to get the results of the model on the training and validation (or testing) set

    Inputs:
        train_dict: dictionary with training features and targets
        valid_dict: dictionary with validation features and targets
        sales_item_qty_model: fitted SalesItemQtyModel

    '''

    #Setting training and validation X and targets
    train_X = train_dict['train_features']
    train_y_sales = train_dict['train_net_sales']
    train_y_item_qty = train_dict['train_item_qty']

    val_X = valid_dict['train_features']
    val_y_sales = valid_dict['train_net_sales']
    val_y_item_qty = valid_dict['train_item_qty']

    #train score
    score_train_sales = sales_item_qty_model.score_sales(train_X, train_y_sales)
    score_train_item_qty = sales_item_qty_model.score_item_qty(train_X, train_y_item_qty)

    #val score
    score_val_sales = sales_item_qty_model.score_sales(val_X, val_y_sales)
    score_val_item_qty = sales_item_qty_model.score_item_qty(val_X, val_y_item_qty)

    print('Train Set Results...\n')
    print(f'MAPE for predicting Sales: {score_train_sales}')
    print(f'MAPE for predicting Item Qty: {score_train_item_qty}\n')

    print('Test Set Results...\n')
    print(f'MAPE for predicting Sales: {score_val_sales}')
    print(f'MAPE for predicting Item Qty: {score_val_item_qty}\n')


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

def recursive_forecasting(historical_df, stores_list, depts_list, production_df, preprocessor = DataPreprocessPipeline(),
                          dual_model = SalesItemQtyModel(), main_cols = ['date_id','item_dept','store','net_sales', 'item_qty']):
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

    historical_df = historical_df[main_cols]

    for current_date in production_df['date_id'].unique():
        for dept in depts_list:
            for store in stores_list:

                relevant_historical_data = historical_df[(historical_df['store'] == store) & (historical_df['item_dept'] == dept)]
                
                current_day_data = production_df[
                    (production_df['date_id'] == current_date) & 
                    (production_df['item_dept'] == dept) & 
                    (production_df['store'] == store)
                ]

                combined_df = pd.concat([relevant_historical_data, current_day_data])
                combined_df_processed = preprocessor.transform(combined_df)

                one_row = combined_df_processed.tail(1).drop(columns = main_cols)

                next_day_sales, next_day_item_qty = dual_model.predict(one_row)

                current_day_data['net_sales'] = next_day_sales
                current_day_data['item_qty'] = next_day_item_qty

                production_df.loc[
                    (production_df['date_id'] == current_date) &
                    (production_df['item_dept'] == dept) &
                    (production_df['store'] == store), ['net_sales', 'item_qty']
                ] = current_day_data[['net_sales', 'item_qty']].values

                historical_df = pd.concat([historical_df, current_day_data], ignore_index=True)

    return production_df


def generate_forecasting_df(start_date, end_date, depts_list, stores_list, historical_df, preprocessor = DataPreprocessPipeline(),
                            dual_model = SalesItemQtyModel(), main_cols = ['date_id','item_dept','store','net_sales', 'item_qty']):
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

    forecasted_df = recursive_forecasting(historical_df=historical_df, stores_list=stores_list,depts_list=depts_list,
                                          production_df=production_df,dual_model=dual_model,preprocessor=preprocessor,
                                          main_cols=main_cols)
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

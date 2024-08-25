import holidays
import pandas as pd
import numpy as np

def preprocess_data(df, date_col = 'date_id', num_lags = 1, rolling_window_size = 2,
                    std_dev = True, use_lag = True, cum_mean = True,cum_sum = True,
                    years = [2021, 2022], return_min = True, return_max = True, week_window_size = 7):
    
    df_process = df.copy()

    df_process = create_time_series_data(df = df_process, date_col=date_col, num_lags=num_lags, rolling_window_size=rolling_window_size,
                                        std_dev=std_dev, use_lag=use_lag, cum_mean=cum_mean, cum_sum=cum_sum,
                                        years=years, return_min=return_min, return_max=return_max, week_window_size=week_window_size)
    
    df_process = df_process.dropna(axis = 0)

    return df_process


def create_time_series_data(df, date_col = 'date_id', num_lags = 1, rolling_window_size = 2,
                    std_dev = True, use_lag = True, cum_mean = True,cum_sum = True,
                    years = [2021, 2022], return_min = True, return_max = True, week_window_size = 7):
    '''
    Function to preprocess data and create features for the sales dataset
    Inputs:
        df: dataframe to preprocess the data
        date_col: col representing the date

    Output:
        dataframe with created features
    '''

    df_process = df.copy()

    df_process[date_col] = pd.to_datetime(df_process[date_col])
    df_process_gb = df_process.groupby([date_col, 'item_dept', 'store'])[['item_qty', 'net_sales']].sum().reset_index()

    #creating lag features
    df_process_gb_lags = create_lag_features(df = df_process_gb, feature_name = 'item_qty', num_lags = num_lags)
    df_process_gb_lags = create_lag_features(df = df_process_gb_lags, feature_name = 'net_sales', num_lags = num_lags)


    df_process_gb_roll = create_rolling_window_features(df = df_process_gb_lags, feature_name = 'item_qty',
                                                        window_size = rolling_window_size, std_dev = std_dev, use_lag = use_lag)
    df_process_gb_roll = create_rolling_window_features(df = df_process_gb_roll, feature_name = 'net_sales',
                                                        window_size = rolling_window_size, std_dev = std_dev, use_lag = use_lag)
    
    df_process_cum = create_cumulative_features(df = df_process_gb_roll, feature_name = 'item_qty',
                                                cum_mean = cum_mean, cum_sum = cum_sum, use_lag = use_lag)
    df_process_cum = create_cumulative_features(df = df_process_cum, feature_name = 'net_sales',
                                                cum_mean = cum_mean, cum_sum = cum_sum, use_lag = use_lag)
    
    df_process_time = create_time_based_features(df = df_process_cum, date_col = date_col, years=years)

    df_process_expand = create_expanding_window_features(df = df_process_time, feature_name = 'item_qty',
                                                        return_min = return_min, return_max = return_max, use_lag = use_lag)
    df_process_expand = create_expanding_window_features(df = df_process_expand, feature_name = 'net_sales',
                                                        return_min = return_min, return_max = return_max, use_lag = use_lag)
    
    df_process_diff = create_daily_weekly_differencing(df = df_process_expand, feature_name = 'item_qty',
                                                       week_window_size = week_window_size, use_lag = use_lag)
    df_process_diff = create_daily_weekly_differencing(df = df_process_diff, feature_name = 'net_sales',
                                                       week_window_size = week_window_size, use_lag = use_lag)
    
    return df_process_diff

def create_lag_features(df, feature_name, num_lags = 1):
    '''
    Creating lag features for the specified field.
    Inputs:
        df: dataframe (sorted by date)
        feature_name: name of the feature to create the lag feature
        num_lags: number of lag features need to be created (default is 1). If more than 1, the specified num of features would be created

    Output:
        dataframe with the created lag features
    '''
    df_process = df.copy()

    for i in range(num_lags):
        df_process[f'lag_{feature_name}_{i+1}'] = df_process.groupby(['item_dept', 'store'])[feature_name].shift(i+1)

    return df_process

def create_rolling_window_features(df, feature_name, window_size = 2, std_dev = True, use_lag = True):
    '''
    Creating rolling window features (mean and, if required, standard deviation) of a feature for a specified window size
    Inputs:
        df: dataframe (sorted by date)
        feature_name: name of the feature to create the rolling window features
        window_size: the length of the previous time steps to consider to create the feature
        std_dev: whether to create the standard deviation feature as well (default is True)
        use_lag: whether to use lag feature to create the feature (default is True). Uses just one lag

    return:
        dataframe with the initial features and the created rolling window features
    '''

    df_process = df.copy()

    if use_lag:
        df_process[f'rolling_mean_{feature_name}_{window_size}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].transform(lambda x: x.rolling(window=window_size).mean())

        if std_dev:
            df_process[f'rolling_std_{feature_name}_{window_size}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].transform(lambda x: x.rolling(window=window_size).std())

        return df_process
    else:
        print("Function not yet designed to use without lag features")
        return None

def create_cumulative_features(df, feature_name, cum_mean = True, cum_sum = True, use_lag = True):
    '''
    Creating cumulative features for the specified field.
    Inputs:
        df: dataframe (sorted by date)
        feature_name: name of the feature to create the cumulative feature
        cum_mean: whether the cumulative mean is required
        cum_sum: whether the cumulative sum is required
        use_lag: whether to use lag feature to create the feature (default is True). Uses just one lag
    
        Output:
            dataframe with the create cumulative feature
    '''

    df_process = df.copy()

    if use_lag:
        if cum_sum:
            df_process[f'cumsum_{feature_name}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cumsum()
        if cum_mean:
            df_process[f'cummean_{feature_name}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cumsum() / df_process.groupby(['store', 'item_dept'])[f'lag_{feature_name}_1'].cumcount()
        
        if cum_mean == False and cum_sum == False:
            print("At least one parameter (cum_sum or cum_mean) should be True")
            return None
        return df_process
    else:
        print("Function not yet designed to use without lag features")
        return None

def create_time_based_features(df, date_col, years = [2021, 2022]):
    '''
    Function to create time based features
    Inputs:
        df: dataframe with date column
        date_col: specify the date column (make sure it is in datetime format)
        years: list of years to get the holidays

    Output:
        dataframe with created features based time
    '''

    df_process = df.copy()

    # Day of the week (0=Monday, 6=Sunday)
    df_process['day_of_week'] = df_process[date_col].dt.dayofweek

    # Is weekend (1=Weekend, 0=Weekday)
    df_process['isWeekend'] = df_process[date_col].dt.dayofweek >= 5
    df_process['isWeekend'] = df_process['isWeekend'].astype(int)

    #Is holiday (1=Holiday, 0=No Holiday). Assuming this store is in US
    us_holidays = holidays.US(years=years)
    df_process['Is_Holiday'] = df_process[date_col].apply(lambda x: x in us_holidays).astype(int)

    return df_process

def create_expanding_window_features(df, feature_name, return_min = True, return_max = True, use_lag = True):
    '''
    Function to create expanding window features (that is get the max and min) while the window expands

    Inputs:
        df: dataframe (sorted by date)
        feature_name: name of feature to create expanding window feature
        return_min: whether to return the minimum across the expanding window
        return_max: whether to return the maximum across the expanding window
        use_lag: whether to use lag feature to create the feature (default is True). Uses just one lag

    Outputs:
        dataframe with created features
    '''

    df_process = df.copy()

    if use_lag:
        if return_min:
            df_process[f'expanding_min_{feature_name}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cummin()
        if return_max:
            df_process[f'expanding_max_{feature_name}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cummax()

        if return_min == False and return_max == False:
            print("At least one parameter (return_min or return_max) should be True")
            return None
        return df_process
    
    else:
        print("Function not yet designed to use without lag features")
        return None


def create_daily_weekly_differencing(df, feature_name, week_window_size = 7, use_lag = True):
    '''
    Function to create daily and weekly difference features (difference between next 2 values or weekly values)
    Inputs:
        df: dataframe (sorted by date)
        feature_name: name of feature to create expanding window feature
        week_window_size: size of the week window (default is 7)
        use_lag: whether to use lag feature to create the feature (default is True). Uses just one lag
    '''

    df_process = df.copy()

    if use_lag:
        df_process[f'diff_{feature_name}'] = df_process.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].diff()

        df_process[f'diff_{feature_name}_{week_window_size}'] = df_process.groupby(['item_dept', 'store'])['item_qty'].diff(week_window_size)

        return df_process
    
    else:
        print("Function not yet designed to use without lag features")
        return None

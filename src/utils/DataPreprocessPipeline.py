import holidays
import pandas as pd

class DataPreprocessPipeline:
    '''
    Data Preprocessing Pipeline with fit, transform, and fit_transform methods.

    This pipeline preprocesses data for time series forecasting by creating lag features, 
    rolling window features, cumulative features, time-based features, expanding window features,
    and daily/weekly difference features.

    Args:
        date_col (str, optional): Name of the date column. Defaults to 'date_id'.
        num_lags (int, optional): Number of lag features to create. Defaults to 1.
        rolling_window_size (int, optional): Size of the rolling window. Defaults to 2.
        std_dev (bool, optional): Whether to calculate rolling standard deviation. Defaults to True.
        use_lag (bool, optional): Whether to use lag features for calculations. Defaults to True.
        cum_mean (bool, optional): Whether to calculate cumulative mean. Defaults to True.
        cum_sum (bool, optional): Whether to calculate cumulative sum. Defaults to True.
        years (list, optional): List of years for holiday calculations. Defaults to [2021, 2022].
        return_min (bool, optional): Whether to calculate expanding minimum. Defaults to True.
        return_max (bool, optional): Whether to calculate expanding maximum. Defaults to True.
        week_window_size (int, optional): Size of the week window. Defaults to 7.
    '''

    def __init__(self, date_col='date_id', num_lags=1, rolling_window_size=2,
                 std_dev=True, use_lag=True, cum_mean=True, cum_sum=True, 
                 years=[2021, 2022], return_min=True, return_max=True, 
                 week_window_size=7):
        
        self.date_col = date_col
        self.num_lags = num_lags
        self.rolling_window_size = rolling_window_size
        self.std_dev = std_dev
        self.use_lag = use_lag
        self.cum_mean = cum_mean
        self.cum_sum = cum_sum
        self.years = years
        self.return_min = return_min
        self.return_max = return_max
        self.week_window_size = week_window_size

    def _create_lag_features(self, df, feature_name):
        '''Creates lag features for a given feature.'''
        for i in range(self.num_lags):
            df[f'lag_{feature_name}_{i+1}'] = df.groupby(['item_dept', 'store'])[feature_name].shift(i+1)
        return df
    
    def _groupby_df(self, df):
        '''Groups the dataframe by the given feature list.'''
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df_gb = df.groupby([self.date_col, 'item_dept', 'store'])[['item_qty', 'net_sales']].sum().reset_index()
        return df_gb

    def _create_rolling_window_features(self, df, feature_name):
        '''Creates rolling window features for a given feature.'''
        if self.use_lag:
            df[f'rolling_mean_{feature_name}_{self.rolling_window_size}'] = (
                df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1']
                .transform(lambda x: x.rolling(window=self.rolling_window_size).mean())
            )
            if self.std_dev:
                df[f'rolling_std_{feature_name}_{self.rolling_window_size}'] = (
                    df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1']
                    .transform(lambda x: x.rolling(window=self.rolling_window_size).std())
                )
        return df

    def _create_cumulative_features(self, df, feature_name):
        '''Creates cumulative features for a given feature.'''
        if self.use_lag:
            if self.cum_sum:
                df[f'cumsum_{feature_name}'] = df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cumsum()
            if self.cum_mean:
                df[f'cummean_{feature_name}'] = (
                    df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cumsum() / 
                    df.groupby(['store', 'item_dept'])[f'lag_{feature_name}_1'].cumcount()
                )
        return df

    def _create_time_based_features(self, df):
        '''Creates time-based features.'''
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['isWeekend'] = (df[self.date_col].dt.dayofweek >= 5).astype(int)
        us_holidays = holidays.US(years=self.years)
        df['Is_Holiday'] = df[self.date_col].apply(lambda x: x in us_holidays).astype(int)
        return df

    def _create_expanding_window_features(self, df, feature_name):
        '''Creates expanding window features for a given feature.'''
        if self.use_lag:
            if self.return_min:
                df[f'expanding_min_{feature_name}'] = (
                    df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cummin()
                )
            if self.return_max:
                df[f'expanding_max_{feature_name}'] = (
                    df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].cummax()
                )
        return df

    def _create_daily_weekly_differencing(self, df, feature_name):
        '''Creates daily and weekly difference features for a given feature.'''
        if self.use_lag:
            df[f'diff_{feature_name}'] = df.groupby(['item_dept', 'store'])[f'lag_{feature_name}_1'].diff()
            df[f'diff_{feature_name}_{self.week_window_size}'] = (
                df.groupby(['item_dept', 'store'])['item_qty'].diff(self.week_window_size)
            )
        return df

    def _preprocess_dataframe(self, df):
        '''Applies all preprocessing steps to a dataframe.'''
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df_gb = df.groupby([self.date_col, 'item_dept', 'store'])[['item_qty', 'net_sales']].sum().reset_index()

        for feature_name in ['item_qty', 'net_sales']:
            df_gb = self._create_lag_features(df_gb, feature_name)
            df_gb = self._create_rolling_window_features(df_gb, feature_name)
            df_gb = self._create_cumulative_features(df_gb, feature_name)
            df_gb = self._create_expanding_window_features(df_gb, feature_name)
            df_gb = self._create_daily_weekly_differencing(df_gb, feature_name)

        df_gb = self._create_time_based_features(df_gb)
        return df_gb.dropna(axis=0)

    def fit(self, X, y=None):
        '''
        Fits the preprocessor to the training data.

        In this case, fit does nothing as we are calculating rolling, expanding, 
        and lag features based on the data itself. We will keep this method for 
        consistency with the sklearn API.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series, optional): The target values. Defaults to None.

        Returns:
            self: Returns the instance itself.
        '''
        return self

    def transform(self, X):
        '''
        Applies the preprocessing steps to a dataframe.

        Args:
            X (pd.DataFrame): The input dataframe to transform.

        Returns:
            pd.DataFrame: The transformed dataframe.
        '''
        return self._preprocess_dataframe(X)

    def fit_transform(self, X, y=None):
        '''
        Fits to the data and then transforms it.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series, optional): The target values. Defaults to None.

        Returns:
            pd.DataFrame: The transformed dataframe.
        '''
        return self.fit(X, y).transform(X)

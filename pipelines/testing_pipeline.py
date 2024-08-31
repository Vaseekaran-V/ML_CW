import pandas as pd
import numpy as np
import sys
import os
import pickle
from pathlib import Path

# Setting path to load util functions
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_dir))

from utils.model_helpers import generate_forecasting_df

def testing_pipeline(data_path, preprocessor_path, model_path, start_date = '2022-02-01', end_date = '2022-02-28'):
    '''
    Function to get the forecasted dataframe when feeding the train data path, preprocessor path, and model_path
    '''

    #Load Data
    df = pd.read_csv(data_path)

    depts_list = df['item_dept'].sort_values().unique()
    stores_list = df['store'].sort_values().unique()

    loaded_preprocessor = pickle.load(open(preprocessor_path, 'rb'))
    trained_model = pickle.load(open(model_path, 'rb'))

    historical_df = loaded_preprocessor._groupby_df(df)

    forecasted_df = generate_forecasting_df(start_date=start_date, end_date=end_date, depts_list=depts_list,
                                            stores_list=stores_list, historical_df=historical_df,
                                            preprocessor=loaded_preprocessor, dual_model=trained_model)
    
    return forecasted_df, historical_df

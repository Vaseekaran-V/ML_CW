from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


class SalesItemQtyModel(BaseEstimator, RegressorMixin):
    '''
    Custom class to handle two models for forecasting two different targets.
    '''
    def __init__(self, model_sales = LinearRegression(), model_item_qty = LinearRegression()):
        self.model_sales = model_sales
        self.model_item_qty = model_item_qty

        self.is_fitted_sales = False
        self.is_fitted_item_qty = False

    def fit(self, X, y_sales, y_item_qty):
        '''
        Fits the model with the given training data and two different targets.

        Inputs:
            X: Training data
            y_sales: Target 1
            y_item_qty: Target 2
        '''
        self.model_sales.fit(X, y_sales)
        self.model_item_qty.fit(X, y_item_qty)

        self.is_fitted_sales = True
        self.is_fitted_item_qty = True

        return self
    
    def predict_sales(self, X):
        '''
        Function to predict sales using sales model

        Inputs:
            X features to predict sales
        
        '''
        if not self.is_fitted_sales:
            raise ValueError('Model not fitted yet. Please fit the model before predicting.')
        else:
            return self.model_sales.predict(X)
        
    def predict_item_qty(self, X):
        '''
        Function to predict item qty using item_qty model

        Inputs:
            X: features to predict item qty
        
        '''
        if not self.is_fitted_item_qty:
            raise ValueError('Model not fitted yet. Please fit the model before predicting.')
        else:
            return self.model_item_qty.predict(X)
        
    def score_sales(self, X, y_sales, scoring_func = mean_absolute_percentage_error):
        '''
        Function to get the score for the sales when predicting the values

        Inputs:
            X: features to predict sales
            y_sales: sales target (ground truth)
            scoring_func: function to calculate the score
        '''
        y_pred = self.predict_sales(X)
        return scoring_func(y_sales, y_pred)
    
    def score_item_qty(self, X, y_item_qty, scoring_func = mean_absolute_percentage_error):
        '''
        Function to get the score for the sales when predicting the values

        Inputs:
            X: features to predict sales
            y_item_qty: item_qty target (ground truth)
            scoring_func: function to calculate the score
        '''
        y_pred = self.predict_item_qty(X)
        return scoring_func(y_item_qty, y_pred)
    
    # def recursive_forecast(self, )

    

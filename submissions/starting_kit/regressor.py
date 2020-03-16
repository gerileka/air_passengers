from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = XGBRegressor(n_estimators= 5000,min_child_weight= 5,max_depth= 30,learning_rate= 0.15,gamma= 0.2,colsample_bytree= 0.3)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

   
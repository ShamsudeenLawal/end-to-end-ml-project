
# import models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import    RandomForestRegressor,\
                                AdaBoostRegressor,\
                                GradientBoostingRegressor

# generating random numbers
from scipy import stats

estimators_configurations = {
    "Linear Regression": {
        "estimator": LinearRegression(),
        "params": {
            "fit_intercept": [True, False]
            }
        },

    "Decision Tree": {
        'estimator': DecisionTreeRegressor(),
        'params': {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best','random'],
            'max_features': ['sqrt','log2', None],
            'max_depth': [None, 1, 3, 5, 7]
            }
    },
    
    "Random Forest": {
        'estimator': RandomForestRegressor(),
        'params': {
            'n_estimators': stats.randint(10, 100,),
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [None, 1, 3, 5, 7],
            "bootstrap": [True, False],
            "max_samples": [0.7, 0.8, 0.9, 1.0,]
        }
    },

    "Gradient Boosting": {
        'estimator': GradientBoostingRegressor(),
        'params': {
            'n_estimators': stats.randint(10, 100,),
            'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1., 10],
            'subsample': [0.7, 0.8, 0.9, 1.0,],
            'criterion': ['squared_error', 'friedman_mse'],
            'max_features': ['auto','sqrt','log2'],
            'max_depth': [None, 1, 3, 5, 7]
        }
    },

    "AdaBoost Regressor": {
        'estimator': AdaBoostRegressor(),
        'params': {
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1., 10],
            'loss': ['linear', 'square', 'exponential'],
            'n_estimators': stats.randint(10, 100,)
        }
    },
    }

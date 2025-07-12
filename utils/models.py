from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LassoLars, RidgeCV
from sklearn.metrics import r2_score

from scipy.stats import median_abs_deviation as mad
from sklearn.model_selection import RandomizedSearchCV


def get_mad_score(y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
                  y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
                  print_version: bool) -> Union[float, None]:
    """
    Calculates the Median Absolute Deviation (MAD) score between true and predicted values.

    Args:
        y_true (Union[pd.Series, pd.DataFrame, np.ndarray]): The true target values.
        y_pred (Union[pd.Series, pd.DataFrame, np.ndarray]): The predicted target values.
        print_version (bool): If True, prints the score; otherwise, returns the score.

    Returns:
        Union[float, None]: The MAD score if `print_version` is False, otherwise None.
    """
    score = mad(abs(y_pred.reshape((-1)) - y_true.reshape(-1)))
    if print_version:
        print(f"MAD score for test set: {score:3e}")
    else:
        return score


def get_r_square_score(y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
                       y_pred: Union[pd.Series, pd.DataFrame, np.ndarray],
                       print_version: bool) -> Union[float, None]:
    score = r2_score(y_true, y_pred)
    """
    Calculates the R-squared score between true and predicted values.

    Args:
        y_true (Union[pd.Series, pd.DataFrame, np.ndarray]): The true target values.
        y_pred (Union[pd.Series, pd.DataFrame, np.ndarray]): The predicted target values.
        print_version (bool): If True, prints the score; otherwise, returns the score.

    Returns:
        Union[float, None]: The R-squared score if `print_version` is False, otherwise None.
    """
    if print_version:
        print(f"R-square score for test set: {score:3e}")
    else:
        return score


# Abstract Base Class for all models
class BaseModel(ABC):
    """
    Abstract Base Class (ABC) for all machine learning models.

    This class defines the common interface that all concrete model implementations
    must adhere to, including methods for fitting, predicting, and evaluating
    training performance.
    """

    def __init__(self):
        """
        Initializes the BaseModel.

        Subclasses should set their specific model instance in `self.model`.
        """
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        """
        Abstract method to fit the model to the training data.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features for training.
            y (Union[pd.Series, np.ndarray]): The target variable for training.
        """
        pass

    @abstractmethod
    def get_train_score(self, X, y):
        """
        Abstract method to calculate and return the training score of the model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features used for training.
            y (Union[pd.Series, np.ndarray]): The true target values used for training.

        Returns:
            float: The training score.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Abstract method to make predictions using the trained model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features for prediction.

        Returns:
            Union[pd.Series, np.ndarray]: The predicted target values.
        """
        pass

class LassoRegressionModel(BaseModel):
    """
    A concrete implementation of BaseModel for Lasso Regression using RandomizedSearchCV.

    This model uses LassoLars and tunes its 'alpha' parameter using RandomizedSearchCV.
    """
    def __init__(self, random_state: int, n_iter: int = 100, cv: int = 5):
        """
        Initializes the model.

        Args:
            random_state (int): The random state for reproducibility.
            n_iter (int): The number of parameter settings that are sampled.
            cv (int): The number of cross-validation folds.
        """
        super().__init__()
        param_dist = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1]}

        lasso = LassoLars(random_state=random_state, max_iter=10000)

        self.model = RandomizedSearchCV(
            estimator=lasso,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )
        self._best_alpha = None

    @property
    def best_alpha(self):
        """Returns the best alpha found by RandomizedSearchCV after fitting."""
        if self._best_alpha is None:
            print("Model has not been fitted yet. The best alpha is not available.")
        return self._best_alpha

    def fit(self, X, y):
        """
        Fits the model to the data using RandomizedSearchCV.

        This method runs the search for the best hyperparameters and then
        refits the model on the entire dataset with those parameters.
        """
        self.model.fit(X, y)
        self._best_alpha = self.model.best_params_['alpha']
        print(f"Randomized search complete. Best alpha found: {self.best_alpha:.4f}")


    def get_train_score(self, X, y, print_version: bool = True):
        """
        Calculates the R-squared score on the training data.

        Note: After fitting, self.model is a RandomizedSearchCV object which
        automatically uses the best estimator for scoring and prediction.
        """
        score = self.model.score(X, y)
        if print_version:
            print(f"Lasso regression R-squared score for train set: {score:.4f}")
            return None
        else:
            return score

    def predict(self, X):
        """Makes predictions using the best-fitted Lasso model."""
        return self.model.predict(X)


class RidgeRegressionModel(BaseModel):
    """
    A concrete implementation of BaseModel for Ridge Regression using RidgeCV.

    This model automatically tunes the alpha parameter using cross-validation.
    """

    def __init__(self, scoring: str = 'r2'):
        """
        Initializes the RidgeRegressionModel.

        Args:
            scoring (str): Strategy to evaluate the performance of the cross-validated model.
                           Defaults to 'r2'.
        """
        super().__init__()
        self.model = RidgeCV(scoring=scoring)
        self.coefficients = None
        self.alpha = None

    def fit(self, X, y):
        """
        Fits the Ridge Regression model to the data.

        The `RidgeCV` model automatically performs cross-validation to find
        the optimal alpha.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features for training.
            y (Union[pd.Series, np.ndarray]): The target variable for training.
        """
        self.model.fit(X, y)
        self.alpha = self.model.alpha_
        self.coefficients = self.model.coef_

    def get_train_score(self, X, y, print_version: bool = True):
        """
        Calculates the R-squared score on the training data for Ridge Regression.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features used for training.
            y (Union[pd.Series, np.ndarray]): The true target values used for training.
            print_version (bool): If True, prints the score; otherwise, returns the score.

        Returns:
            Union[float, None]: The R-squared score if `print_version` is False, otherwise None.
        """
        score = self.model.score(X, y)
        if print_version:
            print(f"Ridge regression R square score for train set: {score:3e}")
            return None
        else:
            return score

    def predict(self, X):
        """
        Makes predictions using the fitted Ridge Regression model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features for prediction.

        Returns:
            Union[pd.Series, np.ndarray]: The predicted target values.
        """
        return self.model.predict(X)


class RandomForestModel(BaseModel):
    """
    A concrete implementation of BaseModel for Random Forest Regression using RandomizedSearchCV.

    This model tunes hyperparameters of a RandomForestRegressor using RandomizedSearchCV.
    """

    def __init__(self, rf_param_dict: dict, random_state: int):
        """
        Initializes the RandomForestModel.

        Args:
            rf_param_dict (dict): Dictionary with parameters names (str) as keys and
                                  distributions or lists of parameters to sample from as values.
            random_state (int): Controls the randomness of the estimator and RandomizedSearchCV.
        """
        super().__init__()
        self.param_dict = rf_param_dict
        self.model = RandomizedSearchCV(RandomForestRegressor(random_state=random_state), rf_param_dict, n_jobs=-1,
                                        random_state=random_state)
        self.best_estimator = None

    def fit(self, X, y):
        """
        Fits the Random Forest model to the data using RandomizedSearchCV.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features for training.
            y (Union[pd.Series, np.ndarray]): The target variable for training.
        """
        self.model.fit(X, y)
        self.best_estimator = self.model.best_estimator_



    def get_train_score(self, print_version: bool = True):
        """
        Retrieves the best score found during RandomizedSearchCV on the training data.

        Note: For RandomizedSearchCV, `best_score_` is typically the mean cross-validated
        score of the best_estimator.

        Args:
            print_version (bool): If True, prints the score; otherwise, returns the score.

        Returns:
            Union[float, None]: The best cross-validation score if `print_version` is False,
                                otherwise None.
        """
        if print_version:
            print(f"Random forest MAD score for train set: {self.model.best_score_:3e}")
            return None
        else:
            return self.model.best_score_


    def predict(self, X):
        """
        Makes predictions using the best-fitted Random Forest model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): The input features for prediction.

        Returns:
            Union[pd.Series, np.ndarray]: The predicted target values.
        """
        return self.best_estimator.predict(X)

    def get_best_estimator_params(self, print_params: bool = True):
        """
        Retrieves the best hyperparameters found by RandomizedSearchCV.

        Args:
            print_params (bool): If True, prints the parameters; otherwise, returns them.

        Returns:
            dict: A dictionary of the best parameters.
        """
        if print_params:
            print(f"Best estimator params are: {self.model.best_params_}")
        return self.model.best_params_


    def get_feature_importance_scores(self):
        """
        Calculates and returns the feature importances and their standard deviations.

        Returns:
            tuple: A tuple containing (feature_importances, feature_importances_std).
        """
        estimation_std = np.std([tree.feature_importances_ for tree in self.best_estimator.estimators_], axis=0)
        return self.best_estimator.feature_importances_, estimation_std

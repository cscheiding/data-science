import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from Tools import preprocessing
import matplotlib.pyplot as plt

def create_and_fit_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocess: bool = True,
        search__cv: int = 5,
        search__scoring: str = 'f1',
        search__n_iter: int = 20,
        search__c_range: tuple[float] = np.logspace(-3, 3, 10)
        ) -> Pipeline:
    """
    - Creates a pipeline that includes:
    1. Pre-processing: if PREPROCESS = TRUE, apply one-hot encoding for
    categorical features and standard scaling for numeric features
    2. Regression: logistic regression
    - A randomized cross-validation search is performed over a range of hyper
    parameters for the logistic regression
    - The resulting best model is fitted to the training data

    Parameters
    -----------
    - X_TRAIN: training split of the features
    - Y_TRAIN: training split of the target
    - PREPROCESS: whether to apply pre-processing to the features, i.e. One-Hot
    encoding for categorical features and Standard Scaling for numeric
    features
    - SEARCH__CV: number of cross-validation folds for the randomized search
    over the logistic regression's hyper parameters
    - SEARCH__SCORING: scoring for the randomized search over the logistic
    regression's hyper parameters 
    - SEARCH__N_ITER: no. of parameter settings that are sampled during the
    randomized search over the logistic regression's hyper parameters
    - SEARCH__C_RANGE: possible values for the C hyper parameter (inverse of
    regularization strength) over which to perform a randomized search

    Returns
    --------
    - BEST_MODEL: pipeline with the cross-validated hyper parameters
    """
    
    # Regression model;
    regression = LogisticRegression()
    # Pipeline that contains the LogisticRegression:
    pipeline = Pipeline([('classifier', regression)])
    # Apply pre-processing to the features' training split:
    if preprocess:
        # ColumnTransfomer that applies One-Hot encoding for categorical
        # features and Standard Scaling for numeric features:
        preprocessor = preprocessing.create_preprocessor(X = X_train)
        # This adds a pre-processing step at the beginning of the pipeline:
        pipeline.steps.insert(0, ('preprocessor', preprocessor))
    # Randomized search on hyper parameters:
    parameters_distributions = {
        'classifier__l1_ratio': (0, 1),
        'classifier__C': search__c_range,
        'classifier__class_weight': (None, 'balanced'),
        'classifier__solver': ('saga', 'liblinear')
    }
    random_search = RandomizedSearchCV(
                            estimator = pipeline,
                            param_distributions = parameters_distributions,
                            n_iter = search__n_iter,
                            scoring = search__scoring,
                            cv = search__cv,
                            n_jobs = -1,
                            random_state = 42
                            )
    # Fit model:
    random_search.fit(X_train, y_train)
    print(f'Results of RandomizedSearchCV')
    print('-' * 30)
    print(f'Best hyper parameters: {random_search.best_params_}')
    print(f'Best {search__scoring} score: {random_search.best_score_}\n')
    # Best model found by the randomized search:
    best_model = random_search.best_estimator_
    return best_model

def show_model_metrics(
                    y_true: np.array,
                    y_pred: np.array,
                    whitespace: int = 15
                    ) -> dict[str: float]:
    """
    - Calculate and display accuracy, precision, recall and F1 of the model that
    predicts the target values Y_PRED, given the real target values Y_TRUE.

    Parameters
    -----------
    - Y_TRUE: real target values
    - Y_PRED: predicted target values
    - WHITESPACE: number of characters between the beginning of the sentence
    and the score value
    """
    # Accuracy score:
    accuracy = accuracy_score(y_true, y_pred)
    # Precision, recall and F1:
    prf_metrics = (lambda y_true, y_pred:
                   precision_recall_fscore_support(y_true, y_pred))
    (_, precision), (_, recall), (_, f1), _ = prf_metrics(y_true, y_pred)

    print(f'Model metrics\n' + '-' * whitespace * 3)
    print('Accuracy:' + ' ' * (whitespace - len('Accuracy'))  + f'{accuracy}')
    print('Precision:' + ' ' * (whitespace - len('Precision'))  + f'{precision}')
    print('Recall:' + ' ' * (whitespace - len('Recall'))  + f'{recall}')
    print('F1:' + ' ' * (whitespace - len('F1'))  + f'{f1}')

def show_confusion_matrix(
                        y_true: np.array,
                        y_pred: np.array,
                        ) -> None:
    """
    Plot the confusion matrix.

    Parameters
    -----------
    - Y_TRUE: real target values
    - Y_PRED: predicted target values
    """
    matrix = confusion_matrix(y_true = y_true, y_pred = y_pred)
    disp = ConfusionMatrixDisplay(
                confusion_matrix = matrix
                )
    disp.plot()
    plt.title('Confusion matrix', fontsize = 18)
    plt.xlabel('Predicted label', fontsize = 14)
    plt.ylabel('True label', fontsize = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
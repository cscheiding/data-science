import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define custom types:
type NaN = float

def calculate_minimum_salary(salary_range: str) -> int | NaN:
    """
    Extract the minimum salary from a SALARY_RANGE.

    Parameters
    -----------
    - SALARY_RANGE: can either be 'NaN' or contain a salary range in the format
    '{MINIMUM_SALARY}-{MAXIMUM_SALARY}'
    """
    # For a non-null value:
    if salary_range != 'Missing':
        # For a salary range:
        if '-' in salary_range:
            min_salary = int(salary_range.split('-')[0])
        # For a specific salary:
        else:
            min_salary = int(salary_range)
    # For a null value:
    else:
        min_salary = np.nan
    return min_salary

def calculate_maximum_salary(salary_range: str) -> int | NaN:
    """
    Extract the maximum salary from a SALARY_RANGE.

    Parameters
    -----------
    - SALARY_RANGE: can either be 'NaN' or contain a salary range in the format
    '{MINIMUM_SALARY}-{MAXIMUM_SALARY}'
    """
    # For a non-null value:
    if salary_range != 'Missing':
        # For a salary range:
        if '-' in salary_range:
            max_salary = int(salary_range.split('-')[1])
        # For a specific salary:
        else:
            max_salary = int(salary_range)
    # For a null value:
    else:
        max_salary = np.nan
    return max_salary

def create_preprocessor(
        X: pd.DataFrame
        ) -> ColumnTransformer:
    """
    Create and return a ColumnTransformer pre-processor that applies:
    1. One-Hot encoding on the categorical features of X
    2. Standard scaling on the numeric features of X
    3. Passthrough on the boolean features of X

    Parameters
    -----------
    - X: DataFrame that contains any subset of rows vs. all the features

    Returns
    --------
    - PREPROCESSOR
    """
    # This will contain the transformers to be passed to ColumnTransformer:
    transformers = []
    # Numeric features: add StandardScaler transformer
    numeric_features = (X.select_dtypes(include = 'number')
                        .columns.to_list())
    numeric_transformer = ('numeric', StandardScaler(), numeric_features)
    transformers.append(numeric_transformer)
    # Categorical (multi-valued) features: add OneHotEncoder transformer
    categorical_features = (X.select_dtypes(include = 'category')
                            .columns.to_list())
    one_hot_encoding_transformer = (
                            'categorical',
                            OneHotEncoder(
                                    handle_unknown = 'ignore',
                                    sparse_output = False
                                    ),
                            categorical_features
                            )
    transformers.append(one_hot_encoding_transformer)
    # Boolean features:
    boolean_features = X.select_dtypes(include = 'bool').columns.to_list()
    passthrough_transformer = ('boolean', 'passthrough', boolean_features)
    transformers.append(passthrough_transformer)
    # Create a ColumnTransfomer that can be directly applied to the features'
    # data:
    preprocessor = ColumnTransformer(
                        transformers = transformers,
                        remainder = 'passthrough' # Keep non-specified features
                        )
    return preprocessor
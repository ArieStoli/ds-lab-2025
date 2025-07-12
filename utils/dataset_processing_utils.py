from pathlib import Path
from typing import Union
import os

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import utils.models


def rename_original_dataset_files(data_path: str):
    """
    Renames specific original dataset files to standardized names within the given data path.
    This function checks for the existence of several predefined Excel and CSV files
    and renames them to more consistent filenames.
    Args:
        data_path (str): The path where the dataset files are located.
    """
    if os.path.isfile(data_path.joinpath('7.1._Drug_seizures_2018-2022.xlsx')):
        os.rename(data_path.joinpath('7.1._Drug_seizures_2018-2022.xlsx'),
                  data_path.joinpath('unodc_drug_seizures.xlsx'))
    if os.path.isfile(data_path.joinpath('10.1._Drug_related_crimes.xlsx')):
        os.rename(data_path.joinpath('10.1._Drug_related_crimes.xlsx'),
                  data_path.joinpath('unodc_drug_crime_counts.xlsx'))
    if os.path.isfile(data_path.joinpath('5.1_Treatment_by_primary_drug_of_use.xlsx')):
        os.rename(data_path.joinpath('10.1._Drug_related_crimes.xlsx'), data_path.joinpath('unodc_drug_treatment.xlsx'))
    if os.path.isfile(data_path.joinpath('Legality_of_cannabis.csv')):
        os.rename(data_path.joinpath('Legality_of_cannabis.csv'), data_path.joinpath('legalization.csv'))

def load_dataset(file_name, encoding: str = 'utf-8', header_row: int = 1):
    """
    Loads a dataset from the 'data' directory. It automatically determines the operating system
    to construct the correct file path and handles both CSV and Excel files.
    Args:
        file_name (str): The name of the file to load (e.g., 'my_data.csv' or 'my_data.xlsx').
        encoding (str, optional): The encoding to use when reading CSV files. Defaults to 'utf-8'.
        header_row (int, optional): The row number to use as the header for Excel files (1-indexed). Defaults to 1.
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    Raises:
        FileNotFoundError: If the specified file does not exist in the data directory."""
    if os.name == 'nt':  # in case of MS Windows
        data_path = Path(f"{os.getcwd()}\\data\\".replace('\\individual_dataset_notebooks', ''))
    else:  # Linux or Mac
        data_path = Path(f"{os.getcwd()}/data/".replace('/individual_dataset_notebooks', ''))

    if file_name.endswith('csv'):
        return pd.read_csv(data_path.joinpath(file_name), encoding=encoding)
    else:
        return pd.read_excel(data_path.joinpath(file_name), header=header_row, engine='calamine')


def convert_gender_to_ratio(df: pd.DataFrame, index_columns: list, value_col: str, total_value: str = 'Total',
                            gender_values: list = ['Females', "Males"]) -> pd.DataFrame:
    """
    Transforms a DataFrame by pivoting gender-specific values to calculate a total value
    and a gender ratio (Females/Males). It handles missing gender categories by filling with 0
    and addresses division by zero for the ratio.
    Args:
        df (pd.DataFrame): The input DataFrame containing gender-disaggregated data.
        index_columns (list): A list of column names to use as index for pivoting (e.g., ['country', 'year', 'drug_group']).
        value_col (str): The name of the column containing the values to be aggregated (e.g., 'number_of_treatments').
        total_value (str, optional): The string representing the total category in the 'gender' column. Defaults to 'Total'.
        gender_values (list, optional): A list of strings representing female and male categories. Defaults to ['Females', 'Males'].

    Returns:
        A new pandas DataFrame with 'total_treatments' and 'gender_ratio' columns.
    """
    # Pivot the table to get 'Total', 'Females', and 'Males' as columns
    df_pivot = df.pivot_table(
        index=index_columns,
        columns='gender',
        values=value_col
    ).reset_index()

    # Rename columns for clarity (remove the 'gender' column name from index)
    df_pivot.columns.name = None

    # Fill any NaN values in 'Total', 'Females', and 'Males' with 0 before calculations
    # This handles cases where a gender category might be missing for a group
    for col in ['Total', 'Females', 'Males']:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
        df_pivot[col] = df_pivot[col].fillna(0)

    # Calculate 'total_treatments' from the 'Total' column
    df_pivot['total_treatments'] = df_pivot['Total']

    # Calculate 'gender_ratio' using numpy.where to handle division by zero
    # If 'Males' is 0, set gender_ratio to -1.0, otherwise calculate Females / Males
    df_pivot['gender_ratio'] = np.where(
        df_pivot['Males'] == 0,
        -1.0,
        df_pivot['Females'] / df_pivot['Males']
    )

    # Replace 0 in 'total_treatments' with -1.0
    df_pivot['total_treatments'] = df_pivot['total_treatments'].replace(0, -1.0)

    # Select and reorder the desired columns
    return df_pivot[['country', 'year', 'drug_group', 'total_treatments', 'gender_ratio']]


def calc_ratio(nom_col: pd.Series, denom_col: pd.Series) -> pd.Series:
    """
    Calculates the ratio of two pandas Series, handling zero and NaN values.
    Zero values in either the numerator or denominator are treated as NaN before calculation.
    Any resulting NaN values in the ratio are then converted to -1.0.
    Args:
        nom_col (pd.Series): The numerator Series.
        denom_col (pd.Series): The denominator Series.
    Returns:
        pd.Series: A Series containing the calculated ratios, with NaN values replaced by -1.0.
    """
    # convert all 0 values to np.nan
    nom_col = nom_col.replace(0, np.nan)
    denom_col = denom_col.replace(0, np.nan)

    # calculate ratio
    ratio = nom_col / denom_col

    # convert all np.nans to -1.0
    ratio = ratio.fillna(-1.0)

    return ratio


def print_discrepancies(df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str):
    """
    Compares the unique values of two specified columns from two different DataFrames
    and prints any discrepancies found.
    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        col1 (str): The name of the column in df1 to compare.
        col2 (str): The name of the column in df2 to compare.
    """
    df1_set = set(df1[col1].unique())
    df2_set = set(df2[col2].unique())
    discrepancies = df1_set - df2_set | df2_set - df1_set
    if len(discrepancies) < 1:
        print("columns {} and {} in dataframes are the same!".format(col1, col2))
    else:
        print("missing values from df1.{}: {}\n\n".format(col1, df1_set - df2_set))
        print("missing values from df2.{}: {}".format(col2, df2_set - df1_set))


def normalize_by_column(df: pd.DataFrame, norm_col: str, cols_to_normalize: Union[dict[str, str], None] = None,
                        cols_to_remove: Union[list[str], None] = []) -> pd.DataFrame:
    """
    Normalizes specified columns in a DataFrame by dividing them by a chosen normalization column.
    Rows with NaN values in the normalization column are dropped.
    The original columns that were normalized are removed, and optionally, other specified columns can also be removed.
    Args:
        df (pd.DataFrame): The input DataFrame.
        norm_col (str): The name of the column to use for normalization.
        cols_to_normalize (Union[dict[str, str], None], optional): A dictionary where keys are the original
            column names to normalize and values are the new column names for the normalized data.
            If None, no columns are normalized. Defaults to None.
        cols_to_remove (Union[list[str], None], optional): A list of additional column names to remove
            from the DataFrame after normalization. Defaults to [].
    Returns:
        pd.DataFrame: A new DataFrame with normalized columns and specified columns removed."""
    copy_df = df.copy()
    # impute null values for the norm column
    copy_df = copy_df.drop(index=copy_df[copy_df[norm_col].isnull()].index)
    for col_i in df.columns:
        if col_i in cols_to_normalize.keys():
            copy_df[cols_to_normalize[col_i]] = df[col_i] / df[norm_col]
            copy_df = copy_df.drop(columns=[col_i])
    if cols_to_remove is not None:
        copy_df = copy_df.drop(columns=cols_to_remove)
    return copy_df


def fill_nans_by_map(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Fills NaN values in specified columns of a DataFrame using a provided mapping.
    Args:
        df (pd.DataFrame): The input DataFrame.
        col_map (dict): A dictionary where keys are column names and values are the
                        fill values for NaN in those respective columns.
    Returns:
        pd.DataFrame: The DataFrame with NaN values filled according to the map.
    """
    for col_i in df.columns:
        if col_i in col_map:
            df[col_i] = df[col_i].fillna(col_map[col_i])
    return df


def calc_shapley(model: utils.models.BaseModel, data: pd.DataFrame) -> np.ndarray:
    """
    Calculates SHAP (SHapley Additive exPlanations) values for a given model and dataset.
    Args:
        model (utils.models.BaseModel): An instance of a model with a `predict` method.
        data (pd.DataFrame): The dataset for which to calculate SHAP values.
    Returns:
        np.ndarray: An array of SHAP values.
    """
    explainer = shap.Explainer(model.predict, data)
    shap_values = explainer(data)
    return shap_values

def create_subset_by_legal_category_for_plot(df: pd.DataFrame, cols_to_add: dict, samples_per_cat: int = 4,
                                             random_state: int = 42) -> pd.DataFrame:

    """
    Creates a subset of the DataFrame by sampling a fixed number of rows per 'legal_category'.
    New columns can be added to the DataFrame before sampling.
    If a category has fewer samples than `samples_per_cat`, all available samples for that category are included.
    Args:
        df (pd.DataFrame): The input DataFrame, expected to have a 'legal_category' column.
        cols_to_add (dict): A dictionary where keys are new column names and values are the
                            data (e.g., Series or list) to be added as new columns to the DataFrame
                            before sampling.
        samples_per_cat (int, optional): The number of samples to take from each 'legal_category'. Defaults to 4.
        random_state (int, optional): Seed for random number generation to ensure reproducibility. Defaults to 42.
    Returns:
        pd.DataFrame: A new DataFrame containing the sampled data from each legal category.
    """
    for col_name, dat in cols_to_add.items():
        df[col_name] = dat

    sampled_data = []
    for lgl_cat in df['legal_category'].unique():
        filtered_df = df[df['legal_category'] == lgl_cat]
        if len(filtered_df) < samples_per_cat:
            sampled_data.append(filtered_df)
        else:
            sampled_data.append(filtered_df.sample(samples_per_cat,
                                                   random_state=random_state,
                                                   replace=True))

    return pd.concat(sampled_data)


def prep_data_for_tsne(df: pd.DataFrame, filtering_query: str = '', numerical_imputation: str = 'median', categorical_imputation: str = 'most_frequent') -> pd.DataFrame:
    """
    Preprocesses the input DataFrame for t-SNE visualization.
    This includes filtering, imputing missing values (numerical and categorical),
    one-hot encoding categorical features, and scaling numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filtering_query (str): A query string to filter the DataFrame (e.g., "year == 2022").
                               If empty, no filtering is applied.
        numerical_imputation (str): Strategy for numerical imputation ('mean', 'median', 'most_frequent', 'constant').
        categorical_imputation (str): Strategy for categorical imputation ('most_frequent', 'constant').

    Returns:
        pd.DataFrame: A scaled DataFrame ready for t-SNE, with original index preserved.
    """

    # Filter the dataframe
    fdf = df.query(filtering_query).copy() if filtering_query else df.copy()

    # Separate numerical and categorical columns
    numerical_cols = fdf.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = list(set(fdf.columns) - set(numerical_cols))

    df_numerical = fdf[numerical_cols]
    df_categorical = fdf[categorical_cols]

    # Impute missing values in numerical features
    numerical_imputer = SimpleImputer(strategy=numerical_imputation)
    # Convert imputed array back to DataFrame with original columns and index
    df_num_imputed = pd.DataFrame(
        numerical_imputer.fit_transform(df_numerical),
        columns=numerical_cols,
        index=fdf.index
    )

    # Impute missing values in categorical features
    categorical_imputer = SimpleImputer(strategy=categorical_imputation)
    X_categorical_imputed = {}
    for col in df_categorical.columns:
        X_categorical_imputed[col] = categorical_imputer.fit_transform(df_categorical[[col]]).flatten().tolist()

    encoded_features_df = pd.DataFrame(X_categorical_imputed)

    # One-hot encode all categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df_categorical)
    encoded_features_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=fdf.index
    )

    # Concatenate imputed numerical and one-hot encoded categorical features
    # Now both are DataFrames, so concatenation will work
    df_combined_features = pd.concat([df_num_imputed, encoded_features_df], axis=1)

    # Perform scaling
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_combined_features)

    # Return as DataFrame with original index, in case it's useful for subsequent steps
    scaled_df = pd.DataFrame(scaled_array, columns=df_combined_features.columns, index=fdf.index)

    return scaled_df

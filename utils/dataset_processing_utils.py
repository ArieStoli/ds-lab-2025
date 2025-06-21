import numpy as np
import pandas as pd


def convert_gender_to_ratio(df: pd.DataFrame, index_columns: list, value_col: str, total_value: str = 'Total',
                            gender_values: list = ['Females', "Males"]) -> pd.DataFrame:
    """
    Transforms a DataFrame containing 'gender' and 'value' columns into 'total' and 'gender_ratio',
    which is the ratio between females and males.

    Args:
        df: A pandas DataFrame with columns 'country', 'year', 'drug_group',
            'gender', and 'number_of_treatments'.

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
    # convert all 0 values to np.nan
    nom_col = nom_col.replace(0, np.nan)
    denom_col = denom_col.replace(0, np.nan)

    # calculate ratio
    ratio = nom_col / denom_col

    # convert all np.nans to -1.0
    ratio = ratio.fillna(-1.0)

    return ratio


def print_discrepancies(df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str):
    df1_set = set(df1[col1].unique())
    df2_set = set(df2[col2].unique())
    discrepancies = df1_set - df2_set | df2_set - df1_set
    if len(discrepancies) < 1:
        print("columns {} and {} in dataframes are the same!".format(col1, col2))
    else:
        print("missing values from df1.{}: {}\n\n".format(col1, df1_set - df2_set))
        print("missing values from df2.{}: {}".format(col2, df2_set - df1_set))
import logging as log
import pandas as pd


logger = log.getLogger(__name__)
log.basicConfig(level=log.ERROR)


def load_csv_data(file_path):
    """
    Loads data from a CSV file and returns a DataFrame.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    DataFrame: DataFrame object containing the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logger.error(f"The specified file does not exist: {file_path}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {e}")
        return None


def remove_rows_with_missing_values(df):
    """
    Removes rows with missing values (NaN) from a DataFrame.

    Args:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: DataFrame with rows containing missing values removed.
    """
    cleaned_df = df.dropna()
    return cleaned_df


def remove_duplicates(df):
    """
    Removes duplicate rows from a DataFrame.

    Args:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: DataFrame with duplicate rows removed.
    """
    cleaned_df = df.drop_duplicates()

    return cleaned_df

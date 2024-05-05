import logging as log
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging section
logger = log.getLogger(__name__)
log.basicConfig(level=log.ERROR)

# Natural Language Toolkit section
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


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


def tokenize_dataframe(df, column_name):
    """
    Tokenizes text in a specified column of a DataFrame.

    Args:
    df (DataFrame): Input DataFrame.
    column_name (str): Name of the column containing text to be tokenized.

    Returns:
    DataFrame: DataFrame with tokenized text added as a new column.
    """
    # Tokenization for each row in column
    df["tokens"] = df[column_name].apply(lambda x: word_tokenize(x))

    return df


def stem_tokens(tokens):
    """
    Stems a list of tokens using Porter Stemmer.

    Args:
    tokens (list of str): List of tokens.

    Returns:
    list of str: Stemmed text.
    """
    # Initialization of PorterStemmer object
    stemmer = PorterStemmer()

    # Stemming for each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return stemmed_tokens


def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.

    Args:
    tokens (list of str): List of tokens.

    Returns:
    list of str: List of tokens with stopwords removed.
    """
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    return filtered_tokens


def vectorize_text(texts):
    """
    Vectorizes a list of texts using TF-IDF.

    Args:
    texts (list of str): List of texts to be vectorized.

    Returns:
    scipy.sparse.csr_matrix: TF-IDF matrix representing the texts.
    """
    # Initialization of TF-IDF object
    vectorizer = TfidfVectorizer()

    # Vectorization of texts
    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix

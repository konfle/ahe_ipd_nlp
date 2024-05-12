import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from utils import load_csv_data, remove_rows_with_missing_values, remove_duplicates, tokenize_dataframe


class NaiveBayesClassifier:
    """A classifier using Naive Bayes algorithm for text classification."""

    def __init__(self, file_path=None, text_column=None, label_column=None):
        """
        Initialize the NaiveBayesClassifier.

        Args:
            file_path (str): Path to the CSV file containing the data.
            text_column (str): Name of the column containing text data.
            label_column (str): Name of the column containing labels.
        """
        self.vectorizer = TfidfVectorizer()
        self.file_path = file_path
        self.text_column = text_column
        self.label_column = label_column

    def train(self):
        """
        Train the Naive Bayes classifier.

        Returns:
            dict: A dictionary containing evaluation metrics and confusion matrix.
        """
        # Load data
        df = load_csv_data(self.file_path)
        if df is None:
            return None

        # Remove rows with missing values
        df = remove_rows_with_missing_values(df)

        # Remove duplicates
        df = remove_duplicates(df)

        # Tokenize text
        df = tokenize_dataframe(df, self.text_column)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df[self.text_column], df[self.label_column],
                                                            test_size=0.2, random_state=42)

        # Vectorize text
        vectorizer = self.vectorizer
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Train the Naive Bayes classifier
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_tfidf, y_train)

        # Predict on test data
        y_pred = self.classifier.predict(X_test_tfidf)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Create dictionary with results
        confusion_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_matrix": {
                "true_positive": conf_matrix[0, 0],
                "false_negative": conf_matrix[0, 1],
                "false_positive": conf_matrix[1, 0],
                "true_negative": conf_matrix[1, 1]
            }
        }

        return confusion_dict

    def predict(self, text):
        """
        Predict the class label for a given text.

        Args:
            text (str): The input text to classify.

        Returns:
            str or None: The predicted class label, or None if prediction fails.
        """
        # Tokenize and vectorize text
        tokenized_text = tokenize_dataframe(pd.DataFrame({self.text_column: [text]}), self.text_column)
        text_column = tokenized_text["tokens"]

        # Join tokens into single text
        text_column = text_column.apply(lambda x: ' '.join(x))

        # Check if vectorizer is fitted
        if not hasattr(self.vectorizer, 'vocabulary_'):
            # If not, fit and transform
            text_tfidf = self.vectorizer.fit_transform(text_column)
        else:
            # If yes, only transform
            text_tfidf = self.vectorizer.transform(text_column)

        # Prediction
        prediction = self.classifier.predict(text_tfidf)
        return prediction[0]

    def save_classifier(self, filename, train_metrics):
        """
        Save the trained classifier to a file using pickle.

        Args:
            filename (str): Name of the file to save the classifier to.
            train_metrics (dict): Dictionary containing evaluation metrics and confusion matrix.
        """
        with open(filename, 'wb') as file:
            pickle.dump((self.classifier, self.vectorizer, self.text_column, self.label_column, train_metrics),
                        file)

    @classmethod
    def load_classifier(cls, filename, file_path):
        """
        Load a trained classifier from a file using pickle.

        Args:
            filename (str): Name of the file to load the classifier from.
            file_path (str): Path to the CSV file containing the data.

        Returns:
            NaiveBayesClassifier: An instance of the NaiveBayesClassifier with the loaded classifier.
            dict: Dictionary containing evaluation metrics and confusion matrix.
        """
        with open(filename, 'rb') as file:
            classifier, vectorizer, text_column, label_column, train_metrics = pickle.load(file)
        nb_classifier = cls(file_path, text_column, label_column)
        nb_classifier.classifier = classifier
        nb_classifier.vectorizer = vectorizer
        return nb_classifier, train_metrics

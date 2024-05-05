import unittest
import pandas as pd
from app.utils import (load_csv_data,
                       remove_rows_with_missing_values,
                       remove_duplicates,
                       stem_tokens)


class TestLoadCSVData(unittest.TestCase):
    def test_load_existing_file(self):
        file_path = 'data/spam_NLP.csv'
        df = load_csv_data(file_path)
        self.assertIsNotNone(df)
        self.assertTrue(isinstance(df, pd.DataFrame))

    def test_load_nonexistent_file(self):
        file_path = 'nonexistent_file.csv'
        df = load_csv_data(file_path)
        self.assertIsNone(df)

    def test_load_corrupted_file(self):
        file_path = 'data/test_csv.csv'
        df = load_csv_data(file_path)
        self.assertIsNone(df)


class TestRemoveRowsWithMissingValues(unittest.TestCase):
    def test_remove_rows_with_missing_values(self):
        data = {'A': [1, 2, None, 4],
                'B': [None, 5, 6, 7],
                'C': [8, None, 10, 11]}
        df = pd.DataFrame(data)
        cleaned_df = remove_rows_with_missing_values(df)
        expected_result = pd.DataFrame({'A': [4],
                                        'B': [7],
                                        'C': [11]})
        cleaned_df = cleaned_df.reset_index(drop=True).astype(int)
        expected_result = expected_result.reset_index(drop=True).astype(int)
        self.assertTrue(cleaned_df.equals(expected_result))


class TestRemoveDuplicates(unittest.TestCase):
    def test_remove_all_duplicates(self):
        data = {'A': [1, 2, 2, 3, 4],
                'B': [5, 6, 6, 7, 8],
                'C': [9, 10, 10, 11, 12]}
        df = pd.DataFrame(data)
        cleaned_df = remove_duplicates(df)
        expected_result = pd.DataFrame({'A': [1, 2, 3, 4],
                                        'B': [5, 6, 7, 8],
                                        'C': [9, 10, 11, 12]})
        cleaned_df = cleaned_df.reset_index(drop=True)
        expected_result = expected_result.reset_index(drop=True)
        self.assertTrue(cleaned_df.equals(expected_result))

    def test_no_duplicates(self):
        data = {'A': [1, 2, 3],
                'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        cleaned_df = remove_duplicates(df)
        self.assertTrue(cleaned_df.equals(df))


class TestStemTokens(unittest.TestCase):
    def test_stem_basic_words(self):
        tokens = ["running", "played", "beautiful"]
        expected_stems = ["run", "play", "beauti"]
        stemmed_text = stem_tokens(tokens)
        self.assertEqual(stemmed_text.split(), expected_stems)

    def test_stem_empty_tokens(self):
        tokens = []
        stemmed_text = stem_tokens(tokens)
        self.assertEqual(stemmed_text, "")

    def test_stem_flexion(self):
        tokens = ["running", "ran", "runs"]
        expected_stems = ["run", "ran", "run"]
        stemmed_text = stem_tokens(tokens)
        self.assertEqual(stemmed_text.split(), expected_stems)

    def test_stem_special_cases(self):
        tokens = ["won't", "friend's"]
        expected_stems = ["won't", "friend'"]
        stemmed_text = stem_tokens(tokens)
        self.assertEqual(stemmed_text.split(), expected_stems)


if __name__ == '__main__':
    unittest.main()

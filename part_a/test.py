import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter

from part_a.utils import *
from part_a.neural_network import *


def test_train_data_alignment():
    """
    Test to ensure the train sparse matrix is correctly matched with the train data CSV.

    :param train_sparse_matrix: SciPy sparse matrix of training data.
    :param train_data_csv: Pandas DataFrame of training data from CSV.
    """
    base_path = "../data"
    train_sparse_matrix = load_train_sparse(
        base_path)  # Your train sparse matrix
    train_data_csv = load_train_csv_df(base_path)  # Your train data CSV

    # Check if dimensions match
    num_users, num_questions = train_sparse_matrix.shape
    assert num_users == len(train_data_csv['user_id'].unique()), \
        "Mismatch in number of users"
    assert num_questions == len(train_data_csv['question_id'].unique()), \
        "Mismatch in number of questions"

    # Create counters for user_id and question_id in the CSV
    user_id_counter = Counter(train_data_csv['user_id'])
    question_id_counter = Counter(train_data_csv['question_id'])

    # Check if every user and question in the matrix is in the CSV
    for user_id in range(num_users):
        assert user_id in user_id_counter, \
            f"User ID {user_id} is missing in CSV"
    for question_id in range(num_questions):
        assert question_id in question_id_counter, \
            f"Question ID {question_id} is missing in CSV"

    print("Test Passed: Train sparse matrix is correctly matched with train data CSV")

def test_metadata_types():
    """
    Test to ensure the metadata columns are of the correct type.
    """
    base_path = "../data"
    student_meta = pd.read_csv(
        os.path.join(base_path, 'student_meta_clean.csv'))
    question_meta = pd.read_csv(os.path.join(base_path, 'question_meta.csv'))
    subject_meta = pd.read_csv(os.path.join(base_path, 'subject_meta.csv'))

    student_meta_cleaned = process_student_metadata(student_meta)
    question_meta_cleaned = process_question_metadata(question_meta,
                                                      subject_meta)
    print(student_meta_cleaned.dtypes)
    print(question_meta_cleaned.dtypes)
    print(question_meta_cleaned.shape)
    print(student_meta_cleaned.shape)

    train_data = load_train_csv_df(base_path)

    train_matrix = load_train_sparse(base_path)
    test_data = load_public_test_csv_to_df(base_path)
    valid_data = load_valid_csv_to_df(base_path)
    print(train_data.shape)
    print(data_df_to_sparse(train_data).shape)
    print(train_matrix.shape)
    # Check if the columns are of the correct type
    print(combine_metadata(train_data, student_meta_cleaned, question_meta_cleaned).shape)
    print(combine_metadata(train_data, student_meta_cleaned, question_meta_cleaned).dtypes)


# how many unique values in question_meta, of subject_id

question_meta = pd.read_csv(os.path.join("../data", 'question_meta.csv'))
print(question_meta['subject_id'].nunique())

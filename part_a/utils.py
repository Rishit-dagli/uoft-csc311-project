from scipy.sparse import load_npz

import numpy as np
import pandas as pd
import csv
import os

def process_student_metadata(student_meta_df):
    """
    Process the student metadata DataFrame.

    :param student_meta_df: Pandas DataFrame containing student metadata
    :return: Processed DataFrame
    """
    # Handle missing values in age
    # Assuming NaN or values <= 7 are to be treated as a separate category
    student_meta_df['age'] = student_meta_df['age'].apply(lambda x: -1 if pd.isna(x) or x <= 7 else x)

    # Normalize age - Simple normalization to range 0-1
    max_age = student_meta_df['age'].max()
    student_meta_df['age'] = student_meta_df['age'] / max_age

    # One-Hot Encode Gender
    # Assuming gender categories are 0 (N/A), 1, 2, etc.
    gender_categories = student_meta_df['gender'].unique()
    for gender in gender_categories:
        column_name = f'gender_{gender}'
        student_meta_df[column_name] = (
                    student_meta_df['gender'] == gender).astype(int)

    # Drop original gender column
    student_meta_df = student_meta_df.drop('gender', axis=1)

    return student_meta_df


def process_question_metadata(question_meta_df, subject_meta_df):
    """
    Process the question metadata DataFrame.

    :param question_meta_df: Pandas DataFrame containing question metadata.
    :param subject_meta_df: Pandas DataFrame containing subject metadata.
    :return: Processed DataFrame
    """
    import ast


    # Convert subject_id list (stored as string) to actual list
    question_meta_df['subject_id'] = question_meta_df['subject_id'].apply(ast.literal_eval)

    # Map subject_id to subject names
    subject_id_to_name = dict(zip(subject_meta_df['subject_id'], subject_meta_df['name']))
    question_meta_df['subjects'] = question_meta_df['subject_id'].apply(lambda ids: [subject_id_to_name[id] for id in ids])

    # One-Hot Encode the subjects
    # Flatten the list of subjects and get unique subject names
    all_subjects = sorted(set([subject for sublist in question_meta_df['subjects'] for subject in sublist]))

    # Create one-hot encoded columns for each subject
    for subject in all_subjects:
        question_meta_df[f'subject_{subject}'] = question_meta_df['subjects'].apply(lambda x: int(subject in x))

    # Drop the original subject_id and subjects columns
    question_meta_df.drop(['subject_id', 'subjects'], axis=1, inplace=True)

    return question_meta_df


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["user_id"].append(int(row[1]))
                data["is_correct"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_train_sparse(root_dir="/data"):
    """ Load the training data as a spare matrix representation.

    :param root_dir: str
    :return: 2D sparse matrix
    """
    path = os.path.join(root_dir, "train_sparse.npz")
    if not os.path.exists(path):
        raise Exception("The specified path {} "
                        "does not exist.".format(os.path.abspath(path)))
    matrix = load_npz(path)
    return matrix


def load_train_csv(root_dir="/data"):
    """ Load the training data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "train_data.csv")
    return _load_csv(path)


def load_train_csv_df(root_dir="/data"):
    """ Load the training data as a pandas DataFrame.


    :param root_dir: str
    :return: A pandas DataFrame
    """
    path = os.path.join(root_dir, "train_data.csv")
    return pd.read_csv(path, usecols=['user_id', 'question_id', 'is_correct'])


def load_valid_csv_to_df(root_dir="/data"):
    """
    Load validation data from a CSV file into a Pandas DataFrame.

    :param path: The path to the validation CSV file.
    :return: A DataFrame containing the validation data.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The validation data file at {path} does not exist.")

    return pd.read_csv(path, usecols=["user_id", "question_id", "is_correct"])


def load_public_test_csv_to_df(root_dir="/data"):
    """
    Load test data from a CSV file into a Pandas DataFrame.

    :param path: The path to the test CSV file.
    :return: A DataFrame containing the test data.
    """
    path = os.path.join(root_dir, "test_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"The test data file at {path} does not exist.")

    return pd.read_csv(path, usecols=["user_id", "question_id", "is_correct"])

def load_valid_csv(root_dir="/data"):
    """ Load the validation data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "valid_data.csv")
    return _load_csv(path)


def load_public_test_csv(root_dir="/data"):
    """ Load the test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "test_data.csv")
    return _load_csv(path)


def load_private_test_csv(root_dir="/data"):
    """ Load the private test data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: an empty list.
    """
    path = os.path.join(root_dir, "private_test_data.csv")
    return _load_csv(path)


def save_private_test_csv(data, file_name="private_test_result.csv"):
    """ Save the private test data as a csv file.

    This should be your submission file to Kaggle.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        user_id: a list of user id.
        question_id: a list of question id.
        is_correct: a list of binary value indicating the correctness of
        (user_id, question_id) pair.
    :param file_name: str
    :return: None
    """
    if not isinstance(data, dict):
        raise Exception("Data must be a dictionary.")
    cur_id = 1
    valid_id = ["0", "1"]
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "is_correct"])
        for i in range(len(data["user_id"])):
            if str(int(data["is_correct"][i])) not in valid_id:
                raise Exception("Your data['is_correct'] is not in a valid format.")
            writer.writerow([str(cur_id), str(int(data["is_correct"][i]))])
            cur_id += 1
    return


def evaluate(data, predictions, threshold=0.5):
    """ Return the accuracy of the predictions given the data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param predictions: list
    :param threshold: float
    :return: float
    """
    if len(data["is_correct"]) != len(predictions):
        raise Exception("Mismatch of dimensions between data and prediction.")
    if isinstance(predictions, list):
        predictions = np.array(predictions).astype(np.float64)
    return (np.sum((predictions >= threshold) == data["is_correct"])
            / float(len(data["is_correct"])))


def sparse_matrix_evaluate(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def sparse_matrix_predictions(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the predictions.

    This function can be used for submitting Kaggle competition.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: list
    """
    predictions = []
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    return predictions

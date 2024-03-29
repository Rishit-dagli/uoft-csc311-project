import pandas as pd

train_file = "../../data/train_data.csv"
valid_file = "../../data/valid_data.csv"
test_file = "../../data/test_data.csv"

train_data = pd.read_csv(train_file)
valid_data = pd.read_csv(valid_file)
test_data = pd.read_csv(test_file)

all_data = pd.concat([train_data, valid_data, test_data])

max_question_id = all_data["question_id"].max()
max_user_id = all_data["user_id"].max()

print(f"Max Question ID: {max_question_id}, Max User ID: {max_user_id}")

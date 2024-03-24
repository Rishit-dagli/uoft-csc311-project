import pandas as pd

# Load question_meta.csv and subject_meta.csv
question_df = pd.read_csv("data/question_meta.csv")
subject_df = pd.read_csv("data/subject_meta.csv")

# Initialize an empty list to store subjects
subjects = []

# Iterate over each row in question_df
for index, row in question_df.iterrows():
    subjects_list = row['subject_id']
    sub = subjects_list[1:-1].split(',')
    sub_list = [int(t.strip()) for t in sub]
    subs = []
    for ids in sub_list:
        subject_name = subject_df.loc[subject_df['subject_id'] == ids, 'name'].values[0]
        subs.append(subject_name)
    subjects.append(subs)

# Add subjects list to question_df
question_df['subjects'] = subjects

# Remove the subject_id column
question_df.drop(columns=['subject_id'], inplace=True)

# Save to the new CSV file
question_df.to_csv("data/question_subject_meta_clean.csv", index=False)

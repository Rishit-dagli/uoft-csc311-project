import pandas as pd
from datetime import datetime

# Load student_meta.csv
df = pd.read_csv("/Users/shivesh/Downloads/project-starter-files-praka150/data/student_meta.csv")

# Fill missing gender with most common value
most_common_gender = df['gender'].mode()[0]
df['gender'] = df['gender'].fillna(most_common_gender)

# Convert data_of_birth to datetime and calculate age
df['data_of_birth'] = pd.to_datetime(df['data_of_birth'], errors='coerce')
current_date = datetime(2020, 1, 1)
df['age'] = (current_date - df['data_of_birth']).dt.days // 365
average_age = df['age'].mean()
df['age'] = df['age'].fillna(round(average_age)).astype(int)

# Fill missing premium_pupil with most common value
most_common_premium_pupil = df['premium_pupil'].mode()[0]
df['premium_pupil'] = df['premium_pupil'].fillna(most_common_premium_pupil)

# Save to student_meta_clean.csv
df[['user_id', 'gender', 'age', 'premium_pupil']].to_csv("/Users/shivesh/Downloads/project-starter-files-praka150/data/student_meta_clean.csv", index=False)

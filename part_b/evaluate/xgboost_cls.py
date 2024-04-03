import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load CSV files into pandas DataFrames
train_data = pd.read_csv("../data/train_data.csv")
valid_data = pd.read_csv("../data/valid_data.csv")
test_data = pd.read_csv("../data/test_data.csv")

# Step 2: Create feature and target variables
X_train = train_data.drop(columns=['is_correct'])
y_train = train_data['is_correct']

X_valid = valid_data.drop(columns=['is_correct'])
y_valid = valid_data['is_correct']

X_test = test_data.drop(columns=['is_correct'])
y_test = test_data['is_correct']

# Step 3: Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBClassifier())
])

# Step 4: Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# Step 5: Evaluate on validation data
y_valid_pred = pipeline.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation Accuracy:", valid_accuracy)

# Step 6: Evaluate on test data
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Step 7: Save the trained model
joblib.dump(pipeline, 'xgboost_model.pkl')

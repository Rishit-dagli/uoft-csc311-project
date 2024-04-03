# To run the evaluation on the four baseline models (Bayesian, Linear Reg, Random Forest, Xgboost)
# Change the names on line 22 manually
# To run evaluations on part_a models, see evaluate_part_a.py

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load CSV files into pandas DataFrames
test_data = pd.read_csv("../data/test_data.csv")

# Step 2: Create feature and target variables
X_test = test_data.drop(columns=['is_correct'])
y_test = test_data['is_correct']

# Step 3: Load the trained model
# To run change the model names to one of the trained baseline models:
# bayesian_regression_model.pkl
# linear_regression_model.pkl
# random_forest_model.pkl
# xgboost_model.pkl
pipeline = joblib.load('xgboost_model.pkl')

# Step 4: Evaluate on test data
y_test_pred = pipeline.predict(X_test)
brk = (max(y_test_pred) + min(y_test_pred)) / 2
y_test_pred = (y_test_pred >= brk).astype(int)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Step 5: Calculate additional classification metrics
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)
tn, fp, fn, tp = conf_matrix.ravel()

false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)
true_negative_rate = tn / (tn + fp)
true_positive_rate = tp / (tp + fn)

print("False Positive Rate:", false_positive_rate)
print("False Negative Rate:", false_negative_rate)
print("True Negative Rate:", true_negative_rate)
print("True Positive Rate:", true_positive_rate)

# Step 6: Print classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

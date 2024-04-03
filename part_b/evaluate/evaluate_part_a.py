# To run the evaluation on the models from part A, and our model from part B
# Change the names on line 26 manually

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import torch

# Step 1: Load CSV files into pandas DataFrames
test_data = pd.read_csv("../data/test_data.csv")

# Step 2: Create feature and target variables
X_test = test_data.drop(columns=['is_correct'])
y_test = test_data['is_correct']

# Step 3: Load the predicted values from the .npz file
# To run change the saved prediction file below, choose from:
# by_item.npz - For kNN predictions by item
# by_user.npz - For kNN predictions by user
# itr.npz - For item response theory model
# nnc.npz - For neural network from subpart (c)
# nnd.npz - For neural network from subpart (d)
# test_preds5.npz - For our model from part B

with np.load('test_preds5.npz') as data:
    try:
        y_test_pred = data['preds']
    except:
        pass
    try:
        y_test_pred = data['pred']
    except:
        pass

# y_test_pred = torch.round(torch.tensor(y_test_pred)).numpy()
# yt = []
# for i in y_test_pred:
#     yt.append(i[0])
print(min(y_test_pred))
print(max(y_test_pred))
# print(y_test_pred.shape)
# print(y_test_pred)
# print(min(yt))
# print(max(yt))

# Step 4: Evaluate on test data
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

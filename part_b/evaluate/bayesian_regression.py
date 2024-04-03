import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
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
    ('model', BayesianRidge())
])

# Step 4: Define custom scorer for rounding predicted probabilities
def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = (y_pred >= 0.5).astype(int)
    return accuracy_score(y_true, y_pred_rounded)

rounded_scorer = make_scorer(rounded_accuracy, greater_is_better=True)

# Step 5: Hyperparameter tuning with grid search
param_grid = {
    'model__alpha_1': [0.0001, 0.001, 0.01],
    'model__alpha_2': [1e-6, 1e-5, 1e-4],
    'model__lambda_1': [1e-6, 1e-5, 1e-4],
    'model__lambda_2': [0.0001, 0.001, 0.01],
    'model__max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring=rounded_scorer)
grid_search.fit(X_train, y_train)

# Step 6: Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Step 7: Evaluate on validation data
valid_accuracy = grid_search.best_score_
print("Validation Accuracy:", valid_accuracy)

# Step 8: Evaluate on test data
y_test_pred = grid_search.predict(X_test)
test_accuracy = rounded_accuracy(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Step 9: Save the trained model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'bayesian_regression_model.pkl')

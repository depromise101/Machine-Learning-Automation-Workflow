# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score
import json

# 1. Load the trained model and test data
print("Loading model and test data...")
model = joblib.load('model.pkl')

# Assuming you saved a separate test set or re-load it
df = pd.read_csv(file_path)


X = df.drop(columns=['label', 'label2', 'device'])

y = df['label2']

# Split the data the same way you did in train.py
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# 2. Make predictions on the test set
print("Evaluating model performance...")
y_pred = model.predict(X_test)

# 3. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"Precision: {precision}")


# 4. Save metrics to a JSON file (the artifact we will publish)
metrics = {
    'accuracy': accuracy,
    'f1_score': f1
}
with open('evaluation_results.json', 'w') as f:
    json.dump(metrics, f)

print("Evaluation results saved as 'evaluation_results.json'.")

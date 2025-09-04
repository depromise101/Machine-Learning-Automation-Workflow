# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Example model
import joblib # To save the model

# 1. Load your dataset (CSV file)
file_path = r"C:\Users\DEPROMISE501\Downloads\waze.csv"
df = pd.read_csv(file_path)

# 2. Prepare the data (e.g., split into features and target)
# Replace 'features' and 'target' with your column names
X = df.drop(columns=['label', 'label2', 'device'])

y = df['label2']



# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)


# 4. Train the ML model
print("Training the model...")
model = RandomForestClassifier(random_state=42)

#  Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [None],
             'max_features': [1.0],
             'max_samples': [1.0],
             'min_samples_leaf': [2],
             'min_samples_split': [2],
             'n_estimators': [300],
             }

#  Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

#  Instantiate the GridSearchCV object
model_cv = GridSearchCV(model, cv_params, scoring=scoring, cv=4, refit='recall')


# 5. fit the model
model.fit(X_train, y_train)

# 5. Save the trained model to a file
print("Saving the trained model...")
joblib.dump(model, 'model.pkl')

print("Model training complete. Model saved as 'model.pkl'.")
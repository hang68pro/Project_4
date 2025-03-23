# Machine Learning analyzing coral data to investigate coral bleaching
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# Load data from https://noaa.maps.arcgis.com/apps/instant/portfolio/index.html?appid=8dbe7b587c1d463bb2ad65e8b65da0d5
# Reef name, location, and important data related to coral bleaching, 
# Maximum Mean Temperature (mean warmest temperature corals typically experience)
# Anything 1°C over this value can cause bleaching.
file_path = '/Users/hangtran/Desktop/Project_4/Fiji (1).csv'
data = pd.read_csv(file_path)

# Preview data
print("Data preview: \n", data.head(2), "\n")
required_columns = [
    'Date', 'Latitude', 'Longitude',
    'Sea_Surface_Temperature', 'HotSpots',
    'Degree_Heating_Weeks',  'Bleaching_Alert_Area'
]
# Check missing columns
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f'Missing columns: {col}')

# Split features (X) and target (y)
X = data.drop(columns = ['Date', 'Bleaching_Alert_Area'])
y = data['Bleaching_Alert_Area']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f'Accuracy score: {accuracy: 0.5f} \n')
print(f'Confusion matrix: \n {confusion}')
print("Classification report:")
print(classification_report(y_test, y_pred))

# The model is performing well with an accuracy of 91%
# We have 5 classes (0, 1, 2, 3, 4) indicating 5 bleaching alert areas 
# For class 0, high precision (93%) and high recall (98%) meaning models capture all true positive case
# For class 1, lower recall suggesting some of instance are misclassfied
# For class 2,3, and 4, overall good precision and recall but their number of samples (support) are low 

# Improvement for this model, address class imbalance and hyperparameter tuning
# Address class imbalance, first check the frequency of each class
print(y.value_counts())
# Class 0 is much larger than the other classes
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(y_train_resampled.value_counts()) #Now, every class is balance

# Fit model, predict and evaluation after SMOTE
rf.fit(X_train_resampled, y_train_resampled)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f'Accuracy score after resample: {accuracy: 0.5f} \n')
print(f'Confusion matrix after resample: \n {confusion}')
print("Classification report after resample:")
print(classification_report(y_test, y_pred))
# Accuracy value is 89% though lower suggesting model is now better at predict other classes, not biased toward class 0 
# SMOTE introduce synthetic values for underepresented classes, forcing the model to learn better
# The recall of class 2 and class 4 is improving meaning model is better at detecting minority class

# Hyperparameter tunning 
from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight':['balanced', None]
}
rf_hyper = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    param_distributions=parameters,
    estimator=rf_hyper,
    cv=3,
    scoring='f1_weighted'
)
random_search.fit(X_train_resampled, y_train_resampled)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(f'Accuracy score after resample & hyperparameters tuning: {accuracy: 0.5f} \n')
print(f'Confusion matrix after resample & hyperparamters tuning: \n {confusion}')
print("Classification report after resample & hyperparameters tuning:")
print(classification_report(y_test, y_pred))

# The accuracy value increase slightly (89.381%), the model performs better after tuning. 
# The confusion matrix shows model performs well across all classes. 
# Class 0 has higher of true positive meaning it is well classified
# Classes 1 and 2 still have noticeable misclassifications, but they are more balanced than before, likely due to the resampling technique.
# Classification Report: Class 0 still performs the best, but classes 1 and 3 show decent recall, meaning they are being identified more frequently.
# Class 4 has a high recall (0.95), meaning the model is very good at detecting it, though precision could still be better (0.76).
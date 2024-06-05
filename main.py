import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('diabetes.csv')

features = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
target = data['Outcome']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

over_sampler = RandomOverSampler(sampling_strategy='auto')
X_resampled, y_resampled = over_sampler.fit_resample(X_train, y_train)

model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_resampled, y_resampled)

poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X_rfe)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_resampled, test_size=0.3, random_state=42)

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'class_weight': ['balanced', None]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train_poly, y_train_poly)

best_lr_model = grid.best_estimator_

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('lr', best_lr_model),
    ('rf', rf),
    ('gb', gb)
], voting='soft')

voting_clf.fit(X_train_poly, y_train_poly)

scores = cross_val_score(voting_clf, X_poly, y_resampled, cv=10)
print("Cross-Validation Accuracy: ", scores.mean())

y_pred = voting_clf.predict(X_test_poly)

accuracy = accuracy_score(y_test_poly, y_pred)
cm = confusion_matrix(y_test_poly, y_pred)
report = classification_report(y_test_poly, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)

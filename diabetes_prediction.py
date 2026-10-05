# ============================================
# DIABETES PREDICTION USING 3 FEATURES
# ============================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------

df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# --------------------------------------------
# 2. Select Features
# --------------------------------------------

X = df[
    [
        "HbA1c_level",
        "blood_glucose_level",
        "bmi"
    ]
]

# Target
y = df["diabetes"]


# --------------------------------------------
# 3. Split Dataset
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------
# 4. Create Improved Random Forest Model
# --------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)


# --------------------------------------------
# 5. Train Model
# --------------------------------------------

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# --------------------------------------------
# 6. Make Predictions
# --------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------
# 7. Check Accuracy
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("Accuracy percentage:", round(accuracy * 100, 2), "%")


# --------------------------------------------
# 8. Classification Report
# --------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

# --------------------------------------------
# 10. Feature Importance
# --------------------------------------------

importance = model.feature_importances_

features = [
    "HbA1c Level",
    "Blood Glucose",
    "BMI"
]

plt.figure(figsize=(7, 5))

plt.bar(features, importance)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Diabetes Prediction")

plt.show()

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Diabetes Prediction - Confusion Matrix")

plt.show()

# --------------------------------------------
# 11. Test a New Person
# --------------------------------------------

HbA1c = float(input("\nEnter HbA1c level: "))
glucose = float(input("Enter blood glucose level: "))
bmi = float(input("Enter BMI: "))


new_person = pd.DataFrame({
    "HbA1c_level": [HbA1c],
    "blood_glucose_level": [glucose],
    "bmi": [bmi]
})


# Prediction
prediction = model.predict(new_person)

# Probability
probability = model.predict_proba(new_person)[0][1]


# --------------------------------------------
# 12. Display Prediction
# --------------------------------------------

if prediction[0] == 1:
    print("\nPrediction: DIABETES")
else:
    print("\nPrediction: NO DIABETES")

print("Diabetes probability:", round(probability * 100, 2), "%")
#Load the data

import pandas as pd                     # for DataFrame handling
from data_prep import load_and_clean_data  # our own function

X, y = load_and_clean_data()

print(" Data loaded")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

#---------------------------------------------------------------------------------------

# Split data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  # 20% for testing
    stratify=y,
    random_state=42
)

print(" Train-test split done")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
#---------------------------------------------------------------------------------------------------------------
#Handle missing values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print(" Missing values handled")
print("X_train_imputed shape:", X_train_imputed.shape)
print("X_test_imputed shape:", X_test_imputed.shape)
#----------------------------------------------------------------------------------------------------------------------
# features Scale 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data (only)
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Transform test data with same scaler
X_test_scaled = scaler.transform(X_test_imputed)

print("\n Feature scaling done")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
#-------------------------------------------------------------------------------------------------------------------------------
# Train ONE model – Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(
    max_iter=1000,        # allow more iterations so it can converge
    class_weight="balanced"  # handle imbalance between 0 and 1
)

log_reg_model.fit(X_train_scaled, y_train)

print("\n Logistic Regression model trained")

#===========================================================================================
# Evaluate the Logistic Regression model

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predict class labels (0 or 1)
y_pred = log_reg_model.predict(X_test_scaled)

# Predict probabilities for class = 1 (Alzheimer)
y_pred_proba = log_reg_model.predict_proba(X_test_scaled)[:, 1]

print("\n Classification Report (Logistic Regression)")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

print("\n Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
#=========================================================================================
print("\n Classification Report (Logistic Regression)")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

print("\n Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------------------------------
# Train SECOND model – Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=300,        # number of trees
    max_depth=12,           # limit depth to reduce overfitting
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

print("\nRandom Forest model trained")


#==============================================================================================
# Evaluate the Random Forest model

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predict class labels (0 or 1)
y_pred_rf = rf_model.predict(X_test_scaled)

# Predict probabilities for class = 1 (Alzheimer)
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n Classification Report (Random Forest)")
print(classification_report(y_test, y_pred_rf))

print("ROC-AUC Score (Random Forest):", roc_auc_score(y_test, y_proba_rf))

print("\n Confusion Matrix (Random Forest)")
print(confusion_matrix(y_test, y_pred_rf))
#=======================================================================================
print("\n Classification Report (Random Forest)")
print(classification_report(y_test, y_pred_rf))

print("ROC-AUC Score (Random Forest):", roc_auc_score(y_test, y_proba_rf))

print("\nConfusion Matrix (Random Forest)")
print(confusion_matrix(y_test, y_pred_rf)) 
#----------------------------------------------------------------------------------------------------------
# Train THIRD model 3 – Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

print("\n Gradient Boosting model trained")
#=============================================================
# Evaluate the Gradient Boosting model
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score


# Predict class labels (0 or 1)
y_pred_gb = gb_model.predict(X_test_scaled)

# Predict probabilities for class = 1 (Alzheimer)
y_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (Gradient Boosting)")
print(classification_report(y_test, y_pred_gb))

accuracy  = accuracy_score(y_test, y_pred_gb)
precision = precision_score(y_test, y_pred_gb)
recall    = recall_score(y_test, y_pred_gb)   # Sensitivity
f1        = f1_score(y_test, y_pred_gb)
roc_auc  = roc_auc_score(y_test, y_proba_gb)

print("\n Individual Metrics")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}  (Sensitivity)")
print(f"F1-score      : {f1:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")

cm = confusion_matrix(y_test, y_pred_gb)
TN, FP, FN, TP = cm.ravel()

print("\n Confusion Matrix")
print(cm)

print("\n Confusion Matrix Terms")
print(f"True Positive  (TP): {TP}")
print(f"True Negative  (TN): {TN}")
print(f"False Positive (FP): {FP}")
print(f"False Negative (FN): {FN}")

#--------------------------------------------------------------------------------------------------------------

# MODEL 4: KNN (K-Nearest Neighbors)

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=7   # you can try 3,5,7 etc.
)

knn_model.fit(X_train_scaled, y_train)

print("\n KNN model trained")

#=================================================================
# Evaluate the Gradient Boosting model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predict class labels (0 or 1)
y_pred = knn_model.predict(X_test_scaled)

# Predict probabilities for class = 1
y_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (MODEL NAME)")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score (MODEL NAME):", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix (MODEL NAME)")
print(confusion_matrix(y_test, y_pred))

#------------------------------------------------------------------------------------------------------------------------------
# MODEL 5: SVM

from sklearn.svm import SVC

svm_model = SVC(
    kernel="rbf",          # non-linear kernel
    probability=True,      # REQUIRED for ROC-AUC
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

print("\n SVM model trained")
#==============================================================
# Evaluate SVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Predict class labels (0 or 1)
y_pred_svm = svm_model.predict(X_test_scaled)

# Predict probabilities for class = 1 (Alzheimer)
y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (SVM)")
print(classification_report(y_test, y_pred_svm))

print("ROC-AUC Score (SVM):", roc_auc_score(y_test, y_proba_svm))

print("\nConfusion Matrix (SVM)")
print(confusion_matrix(y_test, y_pred_svm))

#-------------------------------------------------------------------------------------------------------------------------
# =========================================
# STEP: SAVE FINAL GRADIENT BOOSTING MODEL
# =========================================

import os
import joblib

# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Create models folder if not exists
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "imputer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Save objects
joblib.dump(gb_model, MODEL_PATH)
joblib.dump(imputer, IMPUTER_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n Final Gradient Boosting model saved")
print(" Model path   :", MODEL_PATH)
print(" Imputer path :", IMPUTER_PATH)
print(" Scaler path  :", SCALER_PATH)

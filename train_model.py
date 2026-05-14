import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import resample
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('diabetic_data.csv')
print(f"Original shape: {df.shape}")

# Clean
df.replace('?', np.nan, inplace=True)
drop_cols = ['encounter_id', 'patient_nbr', 'weight',
             'payer_code', 'medical_specialty', 'examide', 'citoglipton']
df.drop(columns=drop_cols, inplace=True)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Target
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Save feature info before encoding
categorical_cols = df.select_dtypes(include='object').columns.tolist()
feature_info = {}

# Get unique values for categorical columns before encoding
for col in categorical_cols:
    feature_info[col] = sorted([str(v) for v in df[col].unique().tolist()])

# Encode
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Balance
df_majority = df[df['readmitted'] == 0]
df_minority = df[df['readmitted'] == 1]
df_majority_down = resample(df_majority, replace=False,
                             n_samples=len(df_minority) * 3, random_state=42)
df_balanced = pd.concat([df_majority_down, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_balanced.drop('readmitted', axis=1)
y = df_balanced['readmitted']

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training models...")

# Train all 3
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb = XGBClassifier(n_estimators=100, random_state=42,
                     use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)

threshold = 0.3

def evaluate(model, name):
    prob = model.predict_proba(X_test)[:,1]
    pred = (prob >= threshold).astype(int)
    return {
        'accuracy': round(accuracy_score(y_test, pred), 4),
        'auc': round(roc_auc_score(y_test, prob), 4),
        'f1': round(f1_score(y_test, pred), 4)
    }

results = {
    'Logistic Regression': evaluate(lr, 'Logistic Regression'),
    'Random Forest': evaluate(rf, 'Random Forest'),
    'XGBoost': evaluate(xgb, 'XGBoost')
}

print("Results:", json.dumps(results, indent=2))

# Save everything
with open('models.pkl', 'wb') as f:
    pickle.dump({
        'lr': lr, 'rf': rf, 'xgb': xgb,
        'le_dict': le_dict,
        'feature_names': feature_names,
        'feature_info': feature_info,
        'results': results,
        'threshold': threshold
    }, f)

print("Models saved to models.pkl")
print("Feature names:", feature_names[:10], "...")

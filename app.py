from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models
with open('models.pkl', 'rb') as f:
    data = pickle.load(f)

lr = data['lr']
rf = data['rf']
xgb = data['xgb']
le_dict = data['le_dict']

# Compatibility fix for saved LogisticRegression models
if not hasattr(lr, "multi_class"):
    lr.multi_class = "auto"

feature_names = data['feature_names']
feature_info = data['feature_info']
results = data['results']
threshold = data['threshold']

MODEL_MAP = {
    'logistic_regression': lr,
    'random_forest': rf,
    'xgboost': xgb
}

MODEL_LABELS = {
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost'
}

@app.route('/')
def index():
    return render_template('index.html',
                           feature_names=feature_names,
                           feature_info=feature_info,
                           results=results)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.json
        model_key = form_data.get('model', 'xgboost')
        model = MODEL_MAP.get(model_key, xgb)

        # Build input row
        input_row = {}
        for feat in feature_names:
            val = form_data.get(feat, 0)
            if feat in le_dict:
                le = le_dict[feat]
                try:
                    val = le.transform([str(val)])[0]
                except:
                    val = 0
            else:
                try:
                    val = float(val)
                except:
                    val = 0
            input_row[feat] = val

        input_df = pd.DataFrame([input_row])
        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob >= threshold)

        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        risk_color = "#e74c3c" if prediction == 1 else "#27ae60"

        # Get all model probabilities for comparison
        all_probs = {}
        for key, m in MODEL_MAP.items():
            p = m.predict_proba(input_df)[0][1]
            all_probs[MODEL_LABELS[key]] = round(float(p) * 100, 1)

        return jsonify({
            'success': True,
            'probability': round(float(prob) * 100, 1),
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'all_probs': all_probs,
            'model_used': MODEL_LABELS.get(model_key, 'XGBoost')
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/metrics')
def metrics():
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

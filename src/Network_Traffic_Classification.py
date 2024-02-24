from flask import Flask, request, render_template
import pandas as pd
from pycaret.anomaly import *
from urllib.parse import quote as url_quote
from pathlib import Path
app = Flask(__name__)


# load pipeline
script_dir = Path(_file_).resolve().parent
model_dir = script_dir.parent / 'Models'
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "network_iforest_pipeline" 
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('Network.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        data = {k: [v] for k, v in data.items()}  # Convert to format for DataFrame
        df_data = pd.DataFrame.from_dict(data)
        
        # Assuming predict_model is already correctly implemented and loaded_model is defined
        predictions = predict_model(loaded_model, data=df_data)

        # Assuming 'Anomaly' is a column in the returned predictions DataFrame
        # Check if the prediction is an attack or not
        if predictions['Anomaly'].values[0] == 1:
            prediction_result = "Attack"
        else:
            prediction_result = "Not Attack"
        
        return render_template('prediction_network.html', prediction=prediction_result)

app.run(host='0.0.0.0')

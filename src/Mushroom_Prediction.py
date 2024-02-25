from flask import Flask, request, render_template
from pycaret.classification import load_model, predict_model
from joblib import load
import pandas as pd
import traceback  
from pathlib import Path

# Initialize Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Load the pre-trained model and pipeline
try:
    # Load the PyCaret model
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / 'Models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "Li_Ting_model"
    loaded_model = load_model(model_path)
    
    # Load the PyCaret model
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / 'Models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "Li_Ting_pipeline"
    loaded_pipeline = load_model(model_path)
    print("Pipeline loaded successfully")
    
except Exception as e:
    print("Error loading model or pipeline:", str(e))
    traceback.print_exc()

# Initialize input_data dictionary to store form values
input_data = {}

def preprocess_input(input_data):
    # Convert input_data to a DataFrame
    processed_data = pd.DataFrame(input_data, index=[0])
    return processed_data

# Define predict function
def make_prediction(input_data):
    try:
        processed_input = preprocess_input(input_data)
        prediction = predict_model(loaded_model, data=processed_input)
        result = prediction.iloc[0]['prediction_label']  
        return result
    except Exception as e:
        print("Prediction Error:", str(e))
        traceback.print_exc()
        return 'Error occurred during prediction'


# Define route for home page
@app.route('/')
def home():
    return render_template("index.html", input_data=input_data)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    global input_data
    input_data.update(request.form.to_dict())  # Update input_data with form values
    prediction_result = make_prediction(input_data)
    return render_template('result.html', prediction_result='Mushroom is {}'.format(prediction_result), input_data=input_data)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

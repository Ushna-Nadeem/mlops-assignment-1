from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    # Render the index.html page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form submission
        data = request.form['data']  # Data should be sent as comma-separated: e.g., "200,100,1,50"
        # Convert the input data into a list of floats
        data = [float(x) for x in data.split(',')]
        prediction = model.predict([data])  # Make prediction using the loaded model
        
        # Interpret the result: 0 = Savory, 1 = Sweet
        result = "Sweet" if prediction[0] == 1 else "Savory"
        
        # Return the result as a JSON response
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

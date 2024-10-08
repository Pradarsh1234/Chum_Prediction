from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(data, label_encoder_geography, label_encoder_gender):
    # Encode categorical features
    data['Geography'] = label_encoder_geography.transform([data['Geography']])[0]
    data['Gender'] = label_encoder_gender.transform([data['Gender']])[0]
    return pd.DataFrame([data])

def initialize_encoders():
    # Fit label encoders with known categories (ensure consistency with your model's training data)
    label_encoder_geography = LabelEncoder()
    label_encoder_gender = LabelEncoder()

    label_encoder_geography.fit(['France', 'Spain', 'Germany'])  # Example categories
    label_encoder_gender.fit(['Male', 'Female'])  # Example categories

    return label_encoder_geography, label_encoder_gender
# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Initialize label encoders for Geography and Gender
label_encoder_geography, label_encoder_gender = initialize_encoders()

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data from the webpage
        input_data = {
            'CreditScore': float(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Tenure': int(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': int(request.form['NumOfProducts']),
            'HasCrCard': int(request.form['HasCrCard']),
            'IsActiveMember': int(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary'])
        }

        # Preprocess the input data
        input_df = preprocess_input(input_data, label_encoder_geography, label_encoder_gender)

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        # Set threshold
        threshold = 0.40
        prediction = (probability >= threshold).astype(int)
        threshold_prediction = (probability >= threshold).astype(int)


        # Return the prediction result
        return render_template('index.html', 
            churn_probability=f'{probability[0]:.2f}',
            prediction_text="Customer Might Exit" if prediction[0] == 1 else "Customer will Not Exit",
            # Use the below line code to check prediction based on threshold
            #threshold_prediction=f"Predicted Churn with threshold {threshold}: {'Exited' if threshold_prediction[0] == 1 else 'Not Exited'}"
        )
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

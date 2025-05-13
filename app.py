from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('student_dropout_dataset.csv')

# Load the trained model and preprocessing objects
classifier = joblib.load('classifier.joblib')
ct = joblib.load('column_transformer.joblib')
scaler = joblib.load('scaler.joblib')

# Function to predict dropout
def predict_dropout(student_id):
    # Check if student ID exists in the dataset
    student_data = data[data['StudentID'] == student_id]
    if student_data.empty:
        return "Student ID not found in the dataset."
    
    # Extract features for the student
    student_features = student_data.drop(columns=['Suspensions'])

    # Encode categorical variables for the student
    student_features_encoded = ct.transform(student_features)

    # Scale features for the student
    student_features_scaled = scaler.transform(student_features_encoded)

    # Make prediction for the student
    prediction = classifier.predict(student_features_scaled)
    return "The student is at risk of dropping out." if prediction[0] else "The student is likely to stay."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        student_id = request.form['student_id']
        try:
            student_id = int(student_id)
            result = predict_dropout(student_id)
        except ValueError:
            result = "Invalid input. Please enter a valid student ID."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

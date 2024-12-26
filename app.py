from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('Animal Health predictions.csv')

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split the dataset into features and target
X = df.drop('HealthStatus', axis=1)
y = df['HealthStatus']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def home():
    # Remove duplicates from animal names using set()
    unique_animals = sorted(set(label_encoders['AnimalName'].inverse_transform(range(len(label_encoders['AnimalName'].classes_)))))

    return render_template('index.html',
                           animals=unique_animals,
                           blood_brain_diseases=sorted(label_encoders['BloodBrainDisease'].classes_),
                           appearance_diseases=sorted(label_encoders['AppearenceDisease'].classes_),
                           general_diseases=sorted(label_encoders['GeneralDisease'].classes_),
                           lung_diseases=sorted(label_encoders['LungDisease'].classes_),
                           abdominal_diseases=sorted(label_encoders['AbdominalDisease'].classes_))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = pd.DataFrame({
            'AnimalName': [request.form['animal_name']],
            'BloodBrainDisease': [request.form['blood_brain_disease']],
            'AppearenceDisease': [request.form['appearance_disease']],
            'GeneralDisease': [request.form['general_disease']],
            'LungDisease': [request.form['lung_disease']],
            'AbdominalDisease': [request.form['abdominal_disease']]
        })

        # Replace "None" with a value that can be handled (e.g., -1)
        none_count = 0
        for column in input_data.columns:
            if input_data[column].values[0] == "None":
                input_data[column] = -1  # Assign a value for "None"
                none_count += 1
            else:
                input_data[column] = label_encoders[column].transform(input_data[column])

        # Logic for a default 'Normal' case when too many fields are 'None'
        if none_count == len(input_data.columns) - 1 or none_count >= 3:
            prediction_label = "normal"
        else:
            # Make prediction using the trained model
            prediction = model.predict(input_data)
            prediction_label = label_encoders['HealthStatus'].inverse_transform(prediction)[0].lower()

        # Determine which image to show based on the prediction
        if prediction_label == "critical":
            image_file = "critical.png"
        else:
            image_file = "normal.png"

        return jsonify({'image': image_file})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

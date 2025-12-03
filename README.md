🍽 Restaurant Rating Prediction – Machine Learning + Streamlit App

This project predicts restaurant ratings based on Price, City, and Cuisine using multiple Machine Learning models.
The final model is deployed using Streamlit, and all ML models are saved using pickle for fast loading.

📁 Project Structure
├── app.py                 # Streamlit web app
├── model.py               # Model training script
├── Dataset.csv            # Training dataset
├── requirements.txt       # Python dependencies
├── runtime.txt            # Python version for deployment
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── label_encoders.pkl

🚀 Features

✔ Trains three ML models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

✔ Automatically detects required dataset columns
✔ Encodes categorical features
✔ Saves trained models + label encoders
✔ Streamlit UI for predicting restaurant ratings
✔ User-friendly dropdown inputs
✔ Option to select prediction model
✔ Fully deployable on Streamlit Cloud / local system

🧠 Model Training

Run the following command to train and save models:

python model.py


This script:

Loads Dataset.csv

Detects columns for price, city, cuisines, rating

Encodes categorical data

Splits into train/test sets

Trains 3 ML models

Saves all models inside the models/ folder

🌐 Run the Streamlit App

To launch the web app locally:

streamlit run app.py


The app includes:

Drop-down selection for Price, City, Cuisine

Option to choose ML model

Instant rating prediction

🧩 Deployment (Streamlit Cloud)

Upload these files to your GitHub repo:

app.py
model.py
Dataset.csv
requirements.txt
runtime.txt
models/


Then deploy using:

➡ https://streamlit.io/cloud

No extra configuration needed.

📦 requirements.txt
pandas
numpy
scikit-learn
streamlit

🐍 Python Version (runtime.txt)
python-3.12

📊 Input & Prediction Example

Input:

Price: “High”

City: “Delhi”

Cuisine: “North Indian”

Model: Random Forest

Output:

⭐ Predicted Rating: 4.3

📝 How It Works

Dataset is cleaned and filtered

Label encoders convert categorical values

Models learn relationships between features & rating

Streamlit app loads saved encoders + models

User inputs are encoded → prediction is generated

🤝 Contributing

Pull requests are welcome!
If you find issues, feel free to open an issue ticket.

📄 License

This project is released under the MIT License.

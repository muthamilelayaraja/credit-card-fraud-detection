Credit Card Fraud Detection System
A machine learning-based Credit Card Fraud Detection System developed using Python, Scikit-learn, Pandas, NumPy, and Streamlit.

The application analyzes credit card transaction data and predicts whether a transaction is legitimate or potentially fraudulent.

📌 Project Overview
Credit card fraud is a major problem in digital financial transactions. This project uses machine learning to identify suspicious transactions based on historical transaction data.

The application provides an interactive web interface where users can:

Upload a transaction dataset
View dataset information
Train a machine learning model
Evaluate model performance
Enter transaction details
Predict whether a transaction is fraudulent
View the estimated fraud probability
🚀 Features
📂 Upload CSV transaction dataset
📊 Display transaction statistics
🤖 Machine learning-based fraud detection
⚖️ Handles imbalanced classes using class weighting
📈 Model accuracy evaluation
📋 Classification report
🔢 Confusion matrix
🔍 Individual transaction prediction
🚨 Fraud detection alert
🌐 Interactive Streamlit web interface
🛠️ Technologies Used
Python 3
Streamlit – Web application framework
Pandas – Data processing
NumPy – Numerical operations
Scikit-learn – Machine learning
Logistic Regression – Classification algorithm
📁 Project Structure
Credit-Card-Fraud-Detection/
│
├── app.py
├── requirements.txt
├── creditcard.csv
└── README.md
📊 Dataset
The application expects a CSV dataset containing transaction features and a target column named:

Class
The Class column should contain:

0 = Legitimate transaction
1 = Fraudulent transaction
Example Dataset
Time	V1	V2	V3	Amount	Class
100	1.2	-0.4	0.8	50.00	0
200	-2.1	1.3	-1.2	900.00	1
300	0.8	0.5	1.1	25.50	0
The application automatically selects numeric features for machine learning.

⚙️ Installation
1. Clone the Repository
git clone <your-repository-url>
2. Navigate to the Project
cd Credit-Card-Fraud-Detection
3. Install Dependencies
pip install -r requirements.txt
▶️ Running the Application
Start the Streamlit application using:

streamlit run app.py
The application will open in your web browser.

🔄 How the System Works
             ┌──────────────────┐
             │  Upload Dataset  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Data Preprocessing│
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Train/Test Split │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Feature Scaling  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Logistic Regression│
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Model Evaluation │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Fraud Prediction │
             └──────────────────┘
🧠 Machine Learning Model
The project uses Logistic Regression as the classification algorithm.

Before training:

Missing values are removed.
Numeric features are selected.
Data is divided into training and testing sets.
Features are standardized using StandardScaler.
Logistic Regression is trained using the training data.
The model uses:

class_weight="balanced"
to give additional importance to the minority fraud class.

📈 Model Evaluation
The application displays:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
For fraud detection, precision and recall are particularly important, because a highly imbalanced dataset can make accuracy alone misleading.

🔍 Transaction Prediction
Users can enter transaction feature values through the Streamlit interface.

The system returns one of two results:

Legitimate Transaction
✅ Transaction appears legitimate.
Potential Fraud
⚠️ Potential Fraud Detected!
The application also displays the estimated fraud probability.

📦 Requirements
The requirements.txt file contains:

streamlit
pandas
numpy
scikit-learn
Install them with:

pip install -r requirements.txt
🖥️ Application Screens
The application contains the following sections:

Dataset Upload
Dataset Preview
Transaction Statistics
Model Performance
Classification Report
Confusion Matrix
Transaction Fraud Prediction
🔮 Future Enhancements
The project can be improved by adding:

Random Forest classification
XGBoost classification
Neural Network model
ROC-AUC curve
Precision-Recall curve
Transaction history
User authentication
Admin dashboard
MySQL database
Real-time transaction monitoring
Email/SMS fraud alerts
Advanced data visualization
Model comparison
Automatic model retraining
⚠️ Disclaimer
This project is intended for educational and demonstration purposes only.

It should not be used as a real-world banking fraud prevention system without additional security, validation, monitoring, and compliance measures.

👨‍💻 Author
Your Name

GitHub: <your-github-profile>
Email: <your-email>
📄 License
This project is created for educational purposes. :::

You can save this directly as README.md in your project folder.

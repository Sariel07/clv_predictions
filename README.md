🧮 CLV Predictions
This project predicts Customer Lifetime Value (CLV) using key business metrics—Recency, Frequency, and Monetary value (RFM). It leverages a machine learning pipeline in Python and offers a user-friendly Gradio interface for interactive predictions.

🔍 Problem Statement
Customer Lifetime Value is a critical metric for businesses to:

Identify high-value customers

Optimize marketing and resource allocation

Drive data-informed decision-making

This project builds a regression-based model to estimate CLV from basic customer data inputs.

🛠️ Tech Stack
Language: Python

Libraries: pandas, scikit-learn, joblib, gradio, etc.

Tools: Jupyter Notebook, VS Code, Git, GitHub

Deployment: Gradio UI (Local execution)

⚙️ How It Works
User Input

Recency: Days since last purchase

Frequency: Number of purchases

Monetary: Total amount spent

Model Prediction
A trained regression model calculates and outputs the predicted CLV score.

Interface
The Gradio front-end enables users to easily interact with the model.

🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/Sariel07/clv_predictions.git
cd clv_predictions
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the Gradio App
bash
Copy
Edit
python app.py
You’ll receive a local link in your browser to start using the CLV prediction tool.

📈 Model Used
Linear Regression

Chosen for its simplicity and interpretability.

Suitable for small to medium datasets with numerical features.

Fast and effective for proof-of-concept.

🧪 Future Enhancements
Incorporate more advanced models (e.g., XGBoost, LightGBM)

Add model evaluation metrics (MAE, RMSE, R²) to the UI

Deploy via web (Streamlit/Flask + Heroku/Render)

Include CSV upload for batch predictions

🙋‍♂️ Author
Sariel B.
GitHub: @Sariel07

📄 License
This project is licensed under the MIT License.
See the LICENSE file for more information.

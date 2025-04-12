
# Heart Disease Prediction

This project was built as part of the CS6301 Machine Learning course. It predicts whether a person is likely to have heart disease using the UCI Heart Disease dataset.

Multiple machine learning models—including Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting, AdaBoost, Decision Tree, MLP, and Bagging—are trained and combined using an ensemble voting mechanism.

The user inputs key health parameters, and the backend returns a prediction based on majority voting over the model outputs.

### Tech Stack
- **Frontend**: HTML + Bootstrap
- **Backend**: Python (Flask)  
- **ML Models**: Scikit-learn models  
- **Data**: UCI Heart Disease Dataset

## Run Locally

Clone the project

```bash
  git clone https://github.com/omerkhathab/heart-disease-prediction.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install flask flask-cors numpy scikit-learn
```

Run the notebook files or place the .pkl model files in the root directory from /data folder

Run the app
```bash
  python backend.py
```
Visit http://127.0.0.1:5000 in your browser

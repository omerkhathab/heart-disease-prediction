from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

app = Flask(__name__, template_folder='.')
CORS(app)

models = {
    "adaboost": pickle.load(open('adaboost.pkl', 'rb')),
    "bagging_classifier": pickle.load(open('bagging_classifier.pkl', 'rb')),
    "decision_tree": pickle.load(open('decision_tree.pkl', 'rb')),
    "gradient_boosting": pickle.load(open('gradient_boosting.pkl', 'rb')),
    "knn": pickle.load(open('knn.pkl', 'rb')),
    "logistic_regression": pickle.load(open('logistic_regression.pkl', 'rb')),
    "mlp_classifier": pickle.load(open('mlp_classifier.pkl', 'rb')),
    "naive_bayes": pickle.load(open('naive_bayes.pkl', 'rb')),
    "random_forest": pickle.load(open('random_forest.pkl', 'rb')),
    "svm": pickle.load(open('svm.pkl', 'rb')),
}
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    features = np.array(data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)

    predictions = {
        name: model.predict_proba(scaled_features)[0, 1] for name, model in models.items()
    }
    return jsonify({"predictions": predictions, "status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
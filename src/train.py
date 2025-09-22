from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import json
from data import load_data, split_data
import numpy as np

def fit_model(X_train, y_train, X_test, y_test):
    """
    Train a Decision Tree Classifier and save the model to a file.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    
    # Calculate model metrics
    train_accuracy = accuracy_score(y_train, dt_classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
    
    # Save feature importance scores
    feature_importance = dt_classifier.feature_importances_.tolist()
    
    # Save model metadata
    metadata = {
        "model_type": "DecisionTreeClassifier",
        "hyperparameters": {
            "max_depth": 3,
            "random_state": 12
        },
        "metrics": {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy)
        },
        "feature_importance": {
            "sepal_length": feature_importance[0],
            "sepal_width": feature_importance[1],
            "petal_length": feature_importance[2],
            "petal_width": feature_importance[3]
        },
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_classes": ["setosa", "versicolor", "virginica"]
    }
    
    # Save model and metadata
    joblib.dump(dt_classifier, "../model/iris_model.pkl")
    with open("../model/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train, X_test, y_test)

import joblib
import numpy as np

# Dictionary mapping class indices to species names
IRIS_SPECIES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

def get_species_name(class_id: int) -> str:
    """
    Get the species name from the class ID.
    Args:
        class_id (int): The class ID (0, 1, or 2)
    Returns:
        str: The species name
    """
    return IRIS_SPECIES.get(class_id, "unknown")

def predict_data(X):
    """
    Predict the class labels and probabilities for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        Tuple containing:
            - numpy.ndarray: Predicted class labels
            - numpy.ndarray: Probability scores for each class
    """
    model = joblib.load("../model/iris_model.pkl")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba

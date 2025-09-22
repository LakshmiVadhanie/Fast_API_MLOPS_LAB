from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, get_species_name
from typing import Dict
import json

app = FastAPI(title=" Iris Classifier API",
             description="API for Iris flower classification with detailed predictions")

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    species_name: str
    species_id: int
    confidence_scores: Dict[str, float]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction, probabilities = predict_data(features)
        species_name = get_species_name(int(prediction[0]))
        
        confidence_scores = {
            "setosa": float(probabilities[0][0]),
            "versicolor": float(probabilities[0][1]),
            "virginica": float(probabilities[0][2])
        }
        
        return IrisResponse(
            species_name=species_name,
            species_id=int(prediction[0]),
            confidence_scores=confidence_scores
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model: metrics and feature importance."""
    try:
        with open("../model/model_metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Could not load model metadata"
        )
    


    

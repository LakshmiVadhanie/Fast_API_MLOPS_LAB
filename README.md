Changes made:

1. main.py:

API response structure changes added:
- species_name: Name of the predicted iris species
- species_id: Numerical ID of the species (0, 1, or 2)
- confidence_scores: Dictionary with probability scores for each species

```python
class IrisResponse(BaseModel):
    species_name: str
    species_id: int
    confidence_scores: Dict[str, float]
```
  
2. predict.py:
- Added functionality to return probability scores using model.predict_proba()
- Added a dictionary mapping for species names:
   - Enhanced predict_data() to return both predictions and probabilities

```python
IRIS_SPECIES = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}
```
3. train.py
Trained the model using the DecisionTreeClassifier

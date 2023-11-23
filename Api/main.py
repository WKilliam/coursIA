import pickle
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

class InputData(BaseModel):
    Category: int
    DayOfWeek: int
    Dates: int
    X: float
    Y: float

# Chargez le dataset d'entraînement initial (loan_data.csv)
data = pd.read_csv('loan_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data[["Category", "DayOfWeek", "Dates", "X", "Y"]],
    test_size=0.2,
    random_state=42
)


@app.get("/model/fit/{name}")
async def load_model_and_fit(name: str):
    with open(f"{name}.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    loaded_model.fit(X_train, y_train)
    with open(f"{name}.pkl", 'wb') as file:
        pickle.dump(loaded_model, file)
    return {"message": f"Model {name} fitted successfully"}

# Endpoint pour prédire avec un modèle
@app.post("/model/predict/{name}")
async def predict_model(name: str, input_data: InputData):
    try:
        with open(f"{name}.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        input_values = [
            input_data.Category, input_data.DayOfWeek, input_data.Dates,
            input_data.X, input_data.Y
        ]
        prediction = loaded_model.predict([input_values])[0]

        return {"prediction": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

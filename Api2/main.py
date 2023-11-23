import pickle

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle une seule fois lors du démarrage de l'application
with open('crime_prediction_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


@app.get("/")
def read_root():
    return {"api": "ok"}


class InputData(BaseModel):
    DayOfWeek: int
    X: float
    Y: float
    Hour: int


@app.post("/predict")
async def predict_model(input_data: InputData):
    try:
        # input_values = [input_data.DayOfWeek, input_data.X, input_data.Y, input_data.Hour]
        input_values = [int(input_data.DayOfWeek), float(input_data.X), float(input_data.Y), int(input_data.Hour)]

        # prediction = loaded_model.predict([input_values])[0]
        prediction = int(loaded_model.predict([input_values])[0])

        return {"prediction": prediction}

        # return {"prediction": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
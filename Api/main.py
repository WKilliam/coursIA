import pickle
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = FastAPI()

# class InputData(BaseModel):
#     Hour : int
#     DayOfWeek: int
#     Dates: int
#     X: float
#     Y: float
#
# # Chargez le dataset d'entraînement initial (train.csv)
# data = pd.read_csv('Api/data/train.csv')
# X_train, X_test, y_train, y_test = train_test_split(
#     data[["Hour", "DayOfWeek", "Dates", "X", "Y"]],
#     test_size=0.2,
#     random_state=42
# )


# @app.get("/model/fit/{name}")
# async def load_model_and_fit(name: str):
#     with open(f"{name}.pkl", 'rb') as file:
#         loaded_model = pickle.load(file)
#     loaded_model.fit(X_train, y_train)
#     with open(f"{name}.pkl", 'wb') as file:
#         pickle.dump(loaded_model, file)
#     return {"message": f"Model {name} fitted successfully"}

# Endpoint pour prédire avec un modèle
@app.post("/model/predict/{name}")
async def predict_model(name: str, input_data: str):
    try:
        # with open(f"{name}.pkl", 'rb') as file:
        #     loaded_model = pickle.load(file)

        # return {"prediction": bool(loaded_model.predict(input_data)[0])}
        # input_values = [
        #     input_data.Hour, input_data.DayOfWeek, input_data.Dates,
        #     input_data.X, input_data.Y
        # ]
        # prediction = loaded_model.predict([input_values])[0]

        # return {"prediction": bool(prediction)}
        return {"prediction": bool(1)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
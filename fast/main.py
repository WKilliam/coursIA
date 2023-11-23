from fastapi import FastAPI
from starlette.exceptions import HTTPException
from ia.iaManager import iaManager
from models.models import Data

app = FastAPI()

ia_manager_instance = iaManager()


@app.post("/predictM")
async def machineLearningPredict(content: Data):
    try:
        predict = ia_manager_instance.predictM(content)
        print(predict)
        return predict
    except Exception as e:
        raise HTTPException(status_code=502, detail="Bad Gateway ou Proxy Error")


@app.post("/predictD")
async def machineLearningPredict(content: Data):
    try:
        return ia_manager_instance.predictM(content)
    except Exception as e:
        raise HTTPException(status_code=502, detail="Bad Gateway ou Proxy Error")

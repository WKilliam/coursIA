from models.models import Data

class iaManager:
    def __init__(self):
        pass

    def predictM(self, data: Data):
        print(data)
        predict = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36]
        return predict

    def predictD(self, data: Data):
        predict = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36]
        return predict

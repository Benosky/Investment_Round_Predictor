from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MonthFeaturesExtractor, DropColumn 
from app.model.model import get_prediction, get_batch_prediction
from app.model.model import __version__ as model_version

# 2. Initialize the app
app = FastAPI(title="Investment Round Prediction API")

# 3. Define the single prediction request payload structure using Pydantic
class InvestmentFeatures(BaseModel):
    numEmps: int
    category: str
    city: str
    state: str
    fundedDate: str
    raisedAmt: float
    raisedCurrency: str


# 4. Create the api home page endpoint
@app.get("/")
def home():
    return {'message': "Welcome to the Investment Rounds Prediction API",
            "model_version":model_version}


# 5. Create the single prediction endpoint
@app.post("/predict")
def predict(data: InvestmentFeatures):
    investment_round = get_prediction(data)
    return investment_round #{"investment_round": investment_round.tolist()}

# 6. Define the batch prediction request payload structure using Pydantic
class MultiInvestments(BaseModel):
    investments:list[InvestmentFeatures]

# 5. Create the batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(data: MultiInvestments):
    investments_rounds = get_batch_prediction(data)
    return investments_rounds
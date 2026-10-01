import pickle
import pandas as pd
import re
from pathlib import Path
import sys
from transformers import MonthFeaturesExtractor, DropColumn 


# Force Gunicorn's __main__ module to recognize the class to prevent errors when dockerizing the model
sys.modules['__main__'].DropColumn = DropColumn
sys.modules['__main__'].MonthFeaturesExtractor = MonthFeaturesExtractor


__version__ = "0.2.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/mlp_model_best-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


def get_prediction(data):
    # Convert input data to the format your model expects
    input_df = pd.DataFrame({
                            'numEmps': data.numEmps,
                            'category': data.category,
                            'city': data.city,
                            'state': data.state,
                            'fundedDate': data.fundedDate,
                            'raisedAmt': data.raisedAmt,
                            'raisedCurrency': data.raisedCurrency
                                        }, index=[0])
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Return result
    return {"prediction": prediction.tolist()}


def get_batch_prediction(data):
    feat_dicts = [ {
                    'numEmps': investment.numEmps,
                    'category': investment.category,
                    'city': investment.city,
                    'state': investment.state,
                    'fundedDate': investment.fundedDate,
                    'raisedAmt': investment.raisedAmt,
                    'raisedCurrency': investment.raisedCurrency} for investment in data.investments]
    feat_index = [x for x in range(len(feat_dicts))]
    input_df = pd.DataFrame(feat_dicts, index=feat_index)

    batch_prediction = model.predict(input_df)
        
    # Return result
    return {"batch_prediction": batch_prediction.tolist()}
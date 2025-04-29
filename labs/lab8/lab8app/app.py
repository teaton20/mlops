from fastapi import FastAPI
import joblib
import pandas as pd

# Create app
app = FastAPI()

# Load model
model = joblib.load("reddit_model_pipeline.joblib")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hey babe, the API is serving joblib realness."}

# Predict endpoint
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"prediction": str(prediction)}
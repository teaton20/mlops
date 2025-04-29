from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load full pipeline (Tfidf + LogisticRegression)
model = joblib.load("reddit_model_pipeline.joblib")

@app.get("/")
def read_root():
    return {"message": "Heyyyyy. The API is serving joblib realness."}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]  # get probs for all classes
    confidence = round(max(proba) * 100, 2)  # highest prob as %
    return {
        "prediction": int(prediction),
        "confidence": f"{confidence}%"
    }
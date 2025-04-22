from metaflow import FlowSpec, step
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        print("🍷 Loading fresh wine data for scoring...")
        # For this lab, we’ll just reload wine data & pretend it's "new"
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        self.new_data = df.sample(20, random_state=42)   # Fake "new data"
        self.next(self.load_model)

    @step
    def load_model(self):
        print("📦 Loading registered model from MLflow...")
        self.model = mlflow.sklearn.load_model("models:/Wine_RF_Model/1")
        self.next(self.predict)

    @step
    def predict(self):
        print("🔮 Making predictions...")
        preds = self.model.predict(self.new_data)
        output = pd.DataFrame(preds, columns=["predicted_label"])
        output.to_csv("../data/predictions.csv", index=False)
        print("💅 Predictions saved to data/predictions.csv")
        self.next(self.end)

    @step
    def end(self):
        print("🎉 Scoring flow complete, queen!")

if __name__ == '__main__':
    ScoringFlow()
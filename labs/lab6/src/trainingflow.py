from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

class TrainingFlow(FlowSpec):

    seed = Parameter('seed', default=42)

    @step
    def start(self):
        print("Ingesting data...")
        self.data = pd.read_csv('../data/wine.csv')
        self.next(self.preprocess)

    @step
    def preprocess(self):
        print("Preprocessing data...")
        self.X = self.data.drop(columns=['label'])
        self.y = self.data['label']
        self.next(self.train_model)

    @step
    def train_model(self):
        print("Training Random Forest...")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.seed
        )
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        self.accuracy = accuracy_score(y_val, preds)
        self.model = model
        print(f"💅 Validation Accuracy: {self.accuracy}")
        self.next(self.register_model)

    @step
    def register_model(self):
        print("📦 Logging & Registering model with MLflow...")
        mlflow.set_experiment("Wine_Training_Metaflow")
        with mlflow.start_run():
            mlflow.log_metric("val_accuracy", self.accuracy)
            joblib.dump(self.model, "best_model.pkl")
            mlflow.sklearn.log_model(self.model, "model", registered_model_name="Wine_RF_Model")
        self.next(self.end)

    @step
    def end(self):
        print(f"🎉 Training complete! Model registered with accuracy: {self.accuracy}")

if __name__ == '__main__':
    TrainingFlow()
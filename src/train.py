import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


class ModelTrainer:
    def __init__(self, X_path="../data/processed/data_labeled.csv", y_path="../data/processed/data_targeted.csv"):
        self.X_path = X_path
        self.y_path = y_path

    def load_data(self):
        X = pd.read_csv(self.X_path)
        y = pd.read_csv(self.y_path)["is_high_risk"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_models(self, X_train, X_test, y_train, y_test):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        for name, model in models.items():
            print(f"\n📊 Training: {name}")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            print(classification_report(y_test, preds))
            print("ROC-AUC:", roc_auc_score(y_test, proba))

            with mlflow.start_run(run_name=name):
                mlflow.log_metric("roc_auc", roc_auc_score(y_test, proba))
                mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
                mlflow.sklearn.log_model(model, "model", registered_model_name=name.replace(" ", "_"))

    def tune_model(self, model, param_grid, X_train, y_train):
        grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
        grid.fit(X_train, y_train)
        return grid.best_estimator_


if __name__ == "__main__":
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    trainer.train_models(X_train, X_test, y_train, y_test)

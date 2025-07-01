import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from mlflow.tracking import MlflowClient


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
            "Logistic_Regression": (
                LogisticRegression(max_iter=1000),
                {'C': [0.1, 1.0, 10.0]}
            ),
            "Random_Forest": (
                RandomForestClassifier(),
                {'n_estimators': [100, 200], 'max_depth': [5, 10]}
            ),
            "Gradient_Boosting": (
                GradientBoostingClassifier(),
                {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
            )
        }
        results = []
        for name, (model, param_grid) in models.items():
            print(f"\n📊 Tuning and Training: {name}")
            best_model = self.tune_model(model, param_grid, X_train, y_train)
            print(best_model)
            preds = best_model.predict(X_test)
            proba = best_model.predict_proba(X_test)[:, 1]

            print(classification_report(y_test, preds))
            print("ROC-AUC:", roc_auc_score(y_test, proba))
            roc_auc = roc_auc_score(y_test, proba)
            acc = accuracy_score(y_test, preds)

        results.append({
            "name": name,
            "model": model,
            "roc_auc": roc_auc,
            "accuracy": acc
        })

        print(f"{name} - ROC AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")
        # Choose best by ROC AUC
        best = max(results, key=lambda x: x["roc_auc"])  # or use x["accuracy"] instead

        print(f"\n🏆 Best Model: {best['name']}")
        print(f"ROC AUC: {best['roc_auc']:.4f}, Accuracy: {best['accuracy']:.4f}")

        with mlflow.start_run(run_name=best["name"]):
            mlflow.log_params(best["model"].get_params())
            mlflow.log_metric("roc_auc", best["roc_auc"])
            mlflow.log_metric("accuracy", best["accuracy"])

            mlflow.sklearn.log_model(
                sk_model=best["model"],
                artifact_path="model",
                registered_model_name=best["name"].replace(" ", "_")
            )
            # Automatically promote best model to Production
            client = MlflowClient()
            latest_version = client.get_latest_versions(best["name"].replace(" ", "_"), stages=["None"])[0].version

            client.transition_model_version_stage(
                name=best["name"].replace(" ", "_"),
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"✅ Model {best['name']} version {latest_version} promoted to Production.")

        pd.DataFrame(results).to_csv("../data/metrics/model_scores.csv", index=False)

    def tune_model(self, model, param_grid, X_train, y_train):
        grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
        grid.fit(X_train, y_train)
        return grid.best_estimator_


if __name__ == "__main__":
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.load_data()
    trainer.train_models(X_train, X_test, y_train, y_test)

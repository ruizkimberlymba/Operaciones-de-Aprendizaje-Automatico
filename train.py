"""Migración del baseline TC5061: ejecutar desde la raíz del repositorio."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "TC5061_Experimentacion_CreditRisk_RuizKimberly"
NUM = ["person_age", "log_person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
CAT = ["loan_grade", "person_home_ownership", "loan_intent", "cb_person_default_on_file"]

def arguments():
    p = argparse.ArgumentParser(description="Entrena y registra el baseline de riesgo crediticio TC5061.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--C", type=float, default=1.0, help="Inversa de regularización L2; debe ser positiva.")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced", help="Compensación del desbalance de clases.")
    p.add_argument("--solver", choices=["lbfgs", "newton-cg"], default="lbfgs", help="Algoritmo de optimización compatible con L2.")
    p.add_argument("--max-iter", type=int, default=1000, help="Máximo de iteraciones del optimizador.")
    p.add_argument("--seed", type=int, default=42, help="Semilla del split estratificado y clasificador.")
    p.add_argument("--features", choices=["all", "decision"], default="all", help="decision excluye grado y tasa; disponibilidad restante pendiente de confirmar.")
    p.add_argument("--data", type=Path, default=ROOT / "data/credit_risk_dataset.csv", help="CSV original de Kaggle, conservado sin modificación.")
    p.add_argument("--tracking-uri", default="sqlite:///" + (ROOT / "tracking/mlflow.db").as_posix(), help="Backend MLflow local; base fuera de carpetas sincronizadas.")
    p.add_argument("--experiment", default=EXPERIMENT, help="Experimento descriptivo que agrupa las configuraciones.")
    p.add_argument("--run-name", default="logreg", help="Nombre descriptivo de ejecución.")
    a = p.parse_args()
    if a.C <= 0 or a.max_iter < 1:
        p.error("C debe ser positiva y max-iter >= 1")
    return a

def load_and_clean(path):
    raw = pd.read_csv(path)
    expected = set(NUM + CAT + ["loan_status", "person_income"]) - {"log_person_income"}
    if set(raw.columns) != expected:
        raise ValueError(f"Esquema inesperado: {list(raw.columns)}")
    if not set(raw.loan_status.unique()).issubset({0, 1}):
        raise ValueError("loan_status debe ser binaria")
    df = raw.drop_duplicates().copy()
    duplicates = len(raw) - len(df)
    before = len(df)
    df = df[df.person_age <= 100].copy()
    inconsistent = df.person_emp_length > (df.person_age - 14)
    df.loc[inconsistent, "person_emp_length"] = np.nan
    if (df.person_income < 0).any():
        raise ValueError("Ingreso negativo no admite log1p")
    df["log_person_income"] = np.log1p(df.person_income)
    df = df.drop(columns="person_income")
    for col in CAT:
        df[col] = df[col].astype("category")
    return df, {"n_raw": len(raw), "duplicates_removed": duplicates, "age_removed": before-len(df), "employment_to_nan": int(inconsistent.sum()), "n_clean": len(df), "raw_columns": list(raw.columns), "usable_timestamp": False}

def preprocessing(features):
    num = [c for c in NUM if features == "all" or c != "loan_int_rate"]
    cat = [c for c in CAT if features == "all" or c != "loan_grade"]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ]), num + cat

def evaluate(model, x, y):
    pred, proba = model.predict(x), model.predict_proba(x)[:, 1]
    metrics = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, proba)}
    return metrics, pred, proba

def main():
    a = arguments()
    (ROOT / "tracking").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(a.tracking_uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(a.experiment) is None:
        client.create_experiment(a.experiment, artifact_location=(ROOT / "tracking/artifacts").as_uri())
    mlflow.set_experiment(a.experiment)
    data_hash = hashlib.sha256(a.data.read_bytes()).hexdigest()
    df, audit = load_and_clean(a.data)
    prep, columns = preprocessing(a.features)
    x = df[columns]
    train_x, test_x, train_y, test_y = train_test_split(x, df.loan_status, test_size=.2, stratify=df.loan_status, random_state=a.seed)
    split_hash = hashlib.sha256(json.dumps({"train": train_x.index.tolist(), "test": test_x.index.tolist()}, sort_keys=True).encode()).hexdigest()
    params = {"C": a.C, "class_weight": a.class_weight, "solver": a.solver, "max_iter": a.max_iter, "seed": a.seed, "features": a.features, "test_size": .2, "n_train": len(train_x), "n_test": len(test_x), "dataset_sha256": data_hash, "split_sha256": split_hash, "imputer": "median+indicator", "scaler": "StandardScaler", "threshold": .5, "model": "LogisticRegression"}
    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tags({"source_notebook": "TC5061_Notebook_Baseline_RuizKimberly.ipynb", "evaluation": "random_stratified_holdout_exploratory", "temporal_validation": "not_measured"})
        model = Pipeline([("prep", prep), ("model", LogisticRegression(C=a.C, class_weight=None if a.class_weight == "none" else "balanced", solver=a.solver, max_iter=a.max_iter, random_state=a.seed))])
        model.fit(train_x, train_y)
        dummy_prep, _ = preprocessing(a.features)
        dummy = Pipeline([("prep", dummy_prep), ("model", DummyClassifier(strategy="most_frequent", random_state=a.seed))]).fit(train_x, train_y)
        metrics, pred, proba = evaluate(model, test_x, test_y)
        train_metrics, _, _ = evaluate(model, train_x, train_y)
        dummy_metrics, _, _ = evaluate(dummy, test_x, test_y)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({"train_"+k:v for k,v in train_metrics.items()})
        mlflow.log_metrics({"dummy_"+k:v for k,v in dummy_metrics.items()})
        mlflow.log_metric("n_iter", int(model.named_steps["model"].n_iter_.max()))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            ConfusionMatrixDisplay.from_predictions(test_y, pred, ax=axes[0], cmap="Blues", display_labels=["No default", "Default"])
            RocCurveDisplay.from_estimator(model, test_x, test_y, ax=axes[1], name="Logistica")
            RocCurveDisplay.from_estimator(dummy, test_x, test_y, ax=axes[1], name="Dummy")
            fig.tight_layout(); fig.savefig(folder / "evaluation.png", dpi=150); plt.close(fig)
            predictions = pd.DataFrame({"source_row": test_x.index, "actual": test_y.to_numpy(), "prediction": pred, "probability": proba})
            predictions.to_csv(folder / "predictions.csv", index=False, float_format="%.17g")
            prediction_hash = hashlib.sha256((folder / "predictions.csv").read_bytes()).hexdigest()
            report = {"run_id": run.info.run_id, "params": params, "metrics": metrics, "dummy_metrics": dummy_metrics, "cleaning": audit, "confusion_matrix": confusion_matrix(test_y,pred).tolist(), "predictions_sha256": prediction_hash, "python": platform.python_version(), "versions": {p:importlib.metadata.version(p) for p in ["mlflow","scikit-learn","pandas","numpy","matplotlib"]}}
            (folder / "run_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (folder / "feature_timing_audit.txt").write_text("Datos sin fecha ni identificador. loan_grade y loan_int_rate requieren verificar que existían antes de la aprobación; el modo decision las excluye como sensibilidad. Ingreso, empleo, vivienda, propósito, monto solicitado e historial requieren una instantánea anterior a la decisión. loan_percent_income debe recalcularse con monto solicitado e ingreso conocido. loan_status es solo etiqueta posterior; nunca predictor. No se midió desempeño futuro. Plan: registrar fecha de solicitud/decisión, origen y disponibilidad de campos, maduración del default y separar cohortes cronológicas sin solapamiento de clientes.", encoding="utf-8")
            mlflow.log_artifacts(tmp, artifact_path="evaluation")
        model_info = mlflow.sklearn.log_model(model, name="model", signature=infer_signature(test_x, model.predict(test_x)), input_example=test_x.head(3), skops_trusted_types=["numpy.dtype"])
        report["model_uri"] = model_info.model_uri
        report["model_id"] = model_info.model_id
        evidence = ROOT / "evidence"; evidence.mkdir(exist_ok=True)
        (evidence / f"{run.info.run_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"run_id": run.info.run_id, "name": a.run_name, "metrics": metrics, "predictions_sha256": prediction_hash, "model_uri": model_info.model_uri}))

if __name__ == "__main__":
    main()

"""Verifica runs terminados, repetibilidad y recuperación del modelo registrado."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from train import ROOT, EXPERIMENT, load_and_clean, preprocessing
from sklearn.model_selection import train_test_split

def main():
    mlflow.set_tracking_uri("sqlite:///"+(ROOT/"tracking/mlflow.db").as_posix())
    log=json.loads((ROOT/"evidence/execution_log.json").read_text(encoding="utf-8"))
    reports=[json.loads((ROOT/"evidence"/(json.loads(c["stdout"].strip())["run_id"]+".json")).read_text(encoding="utf-8")) for c in log]
    assert len(reports)==7 and all(c["returncode"]==0 for c in log)
    a,b=reports[0],reports[5]
    assert log[0]["argv"]==log[5]["argv"]
    assert a["params"]==b["params"] and a["metrics"]==b["metrics"]
    assert a["predictions_sha256"]==b["predictions_sha256"]
    client=mlflow.MlflowClient()
    for r in reports:
        run=client.get_run(r["run_id"])
        assert run.info.status=="FINISHED"
        assert len(run.data.metrics)>=2 and len(client.list_artifacts(r["run_id"],"evaluation"))>=4
    best=max(reports[:5],key=lambda r:r["metrics"]["roc_auc"])
    df,_=load_and_clean(ROOT/"data/credit_risk_dataset.csv")
    _,columns=preprocessing("all")
    _,x,_,y=train_test_split(df[columns],df.loan_status,test_size=.2,stratify=df.loan_status,random_state=42)
    model=mlflow.sklearn.load_model(best["model_uri"])
    path=mlflow.artifacts.download_artifacts(run_id=best["run_id"],artifact_path="evaluation/predictions.csv")
    expected=pd.read_csv(path)
    assert np.array_equal(model.predict(x),expected.prediction)
    assert np.allclose(model.predict_proba(x)[:,1],expected.probability,rtol=0,atol=1e-15)
    rows=[]
    for i,r in enumerate(reports):
        rows.append({"configuration":json.loads(log[i]["stdout"].strip())["name"],"run_id":r["run_id"],**{k:r["params"][k] for k in ["C","class_weight","solver","max_iter","seed","features"]},**r["metrics"]})
    pd.DataFrame(rows).to_csv(ROOT/"evidence/results.csv",index=False)
    expected_data_hash="ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8"
    experiment=client.get_experiment_by_name(EXPERIMENT)
    failed_runs=client.search_runs([experiment.experiment_id],filter_string="attributes.status = 'FAILED'")
    summary={"finished_runs":7,"distinct_model_configurations":5,"identical_arguments":True,"identical_metrics":True,"identical_predictions_sha256":a["predictions_sha256"],"baseline_run_ids":[a["run_id"],b["run_id"]],"best_by_exploratory_holdout_roc_auc":best["run_id"],"logged_model_reload_predictions_match":True,"dataset_sha256":a["params"]["dataset_sha256"],"cleaning":a["cleaning"],"dataset_matches_original":hashlib.sha256((ROOT/"data/credit_risk_dataset.csv").read_bytes()).hexdigest()==expected_data_hash,"temporal_evaluation_performed":False,"initial_failed_run_preserved":bool(failed_runs)}
    (ROOT/"evidence/verification.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()

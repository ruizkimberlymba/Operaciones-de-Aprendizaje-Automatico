# Operaciones-de-Aprendizaje-Automatico
Repository for course work.

## TC5061 Experimentación con MLflow

Python 3.14.3. Dependencias exactas en `requirements.txt`. Dataset público de laotse (Kaggle), CC0; fuente y hash en `data/README.md`.

Desde la raíz del repositorio:

```powershell
python -m pip install -r requirements.txt
python train.py --C 1 --class-weight balanced --solver lbfgs --max-iter 1000 --seed 42 --run-name baseline
```

Para las cinco configuraciones distintas, la repetición y la ablación:

```powershell
python run_experiments.py
python verify_results.py
python -m mlflow server --backend-store-uri sqlite:///tracking/mlflow.db --host 127.0.0.1 --port 5055
```

UI: http://127.0.0.1:5055. Experimento: `TC5061_Experimentacion_CreditRisk_RuizKimberly`. Seed=42; split estratificado 80/20 y preprocesamiento ajustado solo en train. Se registran parámetros, cinco métricas, artifacts y pipeline sklearn. `tracking/` se genera localmente.

`evidence/` conserva resultados y capturas reales. El notebook es solo exploratorio, derivado del original con salidas y celdas de entrenamiento retiradas. La evaluación es aleatoria y exploratoria; no se midió desempeño futuro porque faltan fechas e identificador.

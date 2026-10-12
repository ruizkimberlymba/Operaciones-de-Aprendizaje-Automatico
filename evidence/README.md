# Evidencia

`results.csv` contiene las siete ejecuciones reales utilizadas en el análisis: cinco configuraciones de hiperparámetros, repetición exacta del baseline y ablación de grado/tasa. Las capturas corresponden a la interfaz MLflow original; `evaluation_C100.png` es el artifact registrado del run C_100.

`run_experiments.py` regenera modelos, artifacts y tracking local. `verify_results.py` comprueba igualdad de argumentos, métricas y predicciones de la repetición y recupera el modelo registrado. La auditoría detallada de variables está en `feature_availability_audit.md`.

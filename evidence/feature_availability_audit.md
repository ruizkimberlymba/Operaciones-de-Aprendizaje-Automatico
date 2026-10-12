# Auditoría de disponibilidad de variables

| Campo | Comprobación requerida |
|---|---|
| person_age; person_income; person_emp_length; person_home_ownership | Declarados por el solicitante: verificar captura y validación antes de decidir; no existe fecha que lo pruebe en el CSV. |
| loan_intent; loan_amnt | Confirmar propósito y monto solicitado; un monto finalmente otorgado podría ser posterior a la decisión. |
| loan_percent_income | Recalcular con monto solicitado e ingreso disponible antes de decidir, sin usar valores posteriores. |
| cb_person_default_on_file; cb_person_cred_hist_length | Comprobar fecha de consulta de buró y excluir eventos posteriores a la solicitud. |
| loan_grade; loan_int_rate | Asignados por el prestamista: disponibilidad previa no garantizada. Ablación excluye ambos; no identifica el efecto individual de cada uno. |
| loan_status | Etiqueta posterior usada solo para evaluación/entrenamiento. Nunca predictor. Requiere definir ventana de maduración. |

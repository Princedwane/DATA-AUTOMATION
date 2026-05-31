import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')

runs = mlflow.search_runs(experiment_names=['Titanic_Pipeline'])

print(runs[[
    'start_time',
    'metrics.accuracy',
    'metrics.rows_before',
    'metrics.rows_after',
    'metrics.rows_removed',
    'params.model_type',
    'params.test_size'
]].to_string())

#python check_runs.py to see the metrics
"""MLflow experiment tracking: the training backend, checkpoint models and the eval publisher.

``backend`` holds the logging backend ``MetricsTracker`` builds on rank 0 when
``metrics.enable_mlflow`` is on, and ``worker`` the threads that bound each of its server calls in
time. ``auth`` prepares the environment Databricks auth reads, ``models`` finds or makes the logged
model of a checkpoint, and ``publish`` appends eval metrics to the run a checkpoint was trained
under (``python -m kempnerforge.metrics.mlflow_tracking.publish``). mlflow is imported only by the calls
that need it, so none of these modules requires it at import time.
"""

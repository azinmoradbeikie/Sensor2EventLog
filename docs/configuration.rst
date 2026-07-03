Configuration
=============

All configuration lives in ``config.py`` at the repository root and is organized into a handful of dictionaries.

``PROCESS_STATES``
-------------------

The expert-defined state vocabulary per process type:

.. code-block:: python

   PROCESS_STATES = {
       "production": ["Idle", "Fill", "HeatUp", "Hold", "Cool", "Discharge"],
       "cip": ["PreRinse", "Caustic", "InterRinse", "Acid", "FinalRinse",
               "Sanitize", "Verification", "Standby"],
   }

``FEATURE_CONFIG``
-------------------

Parameters for the abstraction layer's feature families:

.. code-block:: python

   FEATURE_CONFIG = {
       "window_sizes": [5],       # rolling-window sizes for statistical/temporal features
       "stability_eps": 1,        # threshold for the stability flag
       "peak_threshold": 0.1,     # threshold used by peak-related features
   }

``HMM_CONFIG``
---------------

Passed through to the underlying Gaussian HMM:

.. code-block:: python

   HMM_CONFIG = {
       "covariance_type": "diag",
       "n_iter": 100,
       "random_seed": 42,
       "tol": 1e-6,
   }

``DIAGNOSTIC_CONFIG``
-----------------------

Thresholds used by :class:`evaluation.rule_analyzer.RuleDiagnosticAnalyzer` to flag weak rules:

.. code-block:: python

   DIAGNOSTIC_CONFIG = {
       "coverage_threshold": 0.6,
       "precision_threshold": 0.7,
       "explainability_threshold": 0.3,
   }

``VISUALIZATION_CONFIG``
--------------------------

.. code-block:: python

   VISUALIZATION_CONFIG = {
       "gantt_figsize": (14, 8),
       "colors": "Set3",
       "min_duration_for_label": 0.1,   # hours
   }

``PATHS``
----------

Default output locations used by ``main.py``:

.. code-block:: python

   PATHS = {
       "event_log": "pasteurization_event_log.csv",
       "filtered_log": "pasteurization_cleaned_event_log.csv",
       "gantt_chart": "process_gantt_chart.png",
   }

Overriding configuration
--------------------------

Pass any object exposing the same uppercase attributes to
:class:`core.pipeline.Sensor2EventLogPipeline` instead of importing ``config`` directly — this is how the tutorial
walkthrough overrides ``PATHS`` without touching the shared ``config.py`` (see ``tutorial/toy_walkthrough.py``).

Usage
=====

There are two ways to use Sensor2EventLog: the standalone script (``main.py``) for the bundled dataset, or the
``Sensor2EventLogPipeline`` class directly in your own code.

Using the pipeline class
-------------------------

.. code-block:: python

   from core.pipeline import Sensor2EventLogPipeline
   import config

   pipeline = Sensor2EventLogPipeline(config)

   result = pipeline.run(
       data_path="path/to/sensor_data.csv",
       feature_plan=feature_plan,          # see :doc:`feature_plan`
       mode="unsupervised",                # or "supervised"
       use_cip=False,
       n_unsup=None,
       random_seed=42,
       min_duration_seconds=2.0,
       return_intermediate=True,
   )

   event_log = result["event_log"]         # an EventLog object
   model = result["model"]                 # the trained HMM
   predictions = result["predictions"]      # predicted state sequence
   features = result["features"]            # only if return_intermediate=True
   diagnostics = result["diagnostics"]       # only if return_intermediate=True

``run()`` parameters
---------------------

``data_path``
   Path to a CSV file of raw sensor readings.

``feature_plan``
   A dict describing which feature families and rules to compute. See :doc:`feature_plan`.

``mode``
   ``"supervised"`` to seed HMM parameters from labeled data, or ``"unsupervised"`` to fit a vanilla Gaussian HMM
   and align discovered states via the Hungarian algorithm.

``use_cip``
   Whether to include CIP (Clean-In-Place) states alongside production states.

``n_unsup``
   Number of hidden states to fit in unsupervised mode.

``random_seed``
   Seed for reproducibility.

``min_duration_seconds``
   Minimum duration (seconds) for a segment to survive brief-state filtering.

``return_intermediate``
   If ``True``, also returns the engineered ``features`` DataFrame and the rule ``diagnostics`` dict.

Working with the resulting event log
-------------------------------------

.. code-block:: python

   event_log.to_csv("event_log.csv")
   event_log.to_xes("event_log.xes")
   pm4py_log = event_log.to_pm4py()

   filtered = event_log.filter_duration(min_seconds=2.0)
   stats = event_log.get_statistics()

Using the script entry point
------------------------------

For a quick end-to-end run against the bundled pasteurization dataset, use:

.. code-block:: bash

   python3 main.py

This calls the same pipeline under the hood, using the feature plan and paths defined in ``config.py``, and writes
``event_log.csv``, ``filtered_log.csv``, ``gantt_chart.png``, and ``results.txt``.

Tutorial
========

A self-contained, minimal walkthrough is available in the ``tutorial/`` directory of the repository. It generates a
small synthetic sensor dataset, extracts features, evaluates rule diagnostics, and produces an interval event log —
without needing any external data files.

Running it
-----------

.. code-block:: bash

   python3 tutorial/toy_walkthrough.py

This is also available as a Jupyter notebook, ``tutorial/toy_walkthrough.ipynb``, which mirrors the script and
includes a repo-root bootstrap cell for reliable imports when launched from ``tutorial/``. To run it, make sure
``ipykernel`` is installed in your environment:

.. code-block:: bash

   python -m pip install ipykernel

What it produces
------------------

Running the walkthrough writes the following to ``tutorial/output/``:

* ``toy_sensor_data.csv`` — the synthetic raw sensor dataset.
* ``toy_features.csv`` — the engineered feature table.
* ``toy_event_log.csv`` — the resulting interval event log.

It also prints rule-diagnostic recommendations to the console and (if ``matplotlib`` is installed) renders sensor
time-series plots and a simple process graph.

Further tutorials
-------------------

A tiered set of tutorials — from core concepts through advanced process-mining integration and eventization-quality
evaluation — is in progress in the repository's ``tutorials/`` directory. See the `tutorials table in the README
<https://github.com/azinmoradbeikie/Sensor2EventLog#tutorials>`_ for current status.

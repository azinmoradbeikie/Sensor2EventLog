Quickstart
==========

This page walks you through your first run in under five minutes using the
included toy example.

Run the toy walkthrough
-----------------------

After installing the requirements (see :doc:`installation`), run:

.. code-block:: bash

   python3 tutorial/toy_walkthrough.py

This executes the full pipeline on a small synthetic dataset and produces:

* an event log (``event_log.csv``)
* a filtered event log (``filtered_log.csv``)
* a Gantt chart visualization (``gantt_chart.png``)
* a text summary of the run (``results.txt``)

Run the main pipeline
---------------------

To run the main pipeline on the bundled pasteurization dataset:

.. code-block:: bash

   python3 main.py

The default configuration in ``main.py`` uses an unsupervised HMM and the
following feature plan:

.. code-block:: python

   feature_plan = {
       'statistical': ['T', 'Q_in', 'Q_out'],
       'temporal':    ['T', 'Q_in', 'Q_out'],
       'stability':   ['T', 'Q_in', 'Q_out'],
       'interaction': [['T', 'Q_in', 'Q_out']],
       'event': [
           '(T_diff_smooth > 1)',
           '(T_diff_smooth < -1)',
           '(Q_out > 0.3)',
           '(T > 70) & (T_stable_flag == 1)',
           '(Q_in > 0.3) AND (T_diff < 0.2)'
       ],
       'contextual': []
   }

See :doc:`feature_plan` for the full DSL reference.

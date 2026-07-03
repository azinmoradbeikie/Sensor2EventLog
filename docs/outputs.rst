Outputs
=======

Running the pipeline (via ``main.py`` or ``Sensor2EventLogPipeline.run``) produces the following artifacts.

Event log (``event_log.csv``)
-------------------------------

The raw interval event log with one row per decoded activity segment:

.. list-table::
   :header-rows: 1

   * - Column
     - Description
   * - ``case_id``
     - Identifier of the batch/case the segment belongs to.
   * - ``activity``
     - The decoded process state / activity name.
   * - ``start_timestamp`` / ``end_timestamp``
     - Segment boundaries.
   * - ``duration_seconds``
     - Segment length.
   * - ``event_count``
     - Number of raw timestamps folded into the segment.

Filtered event log (``filtered_log.csv``)
--------------------------------------------

The same log with segments shorter than ``min_duration_seconds`` removed, to eliminate spurious flickers between
states before feeding the log into process discovery.

Gantt chart (``gantt_chart.png``)
------------------------------------

A per-case Gantt-style visualization of activity segments over time, generated with the settings in
``config.VISUALIZATION_CONFIG``.

Results summary (``results.txt``)
------------------------------------

A plain-text capture of everything printed during the run — feature computation, rule diagnostics, evaluation
metrics — useful for keeping a record of a specific pipeline run alongside its outputs.

Other export formats
----------------------

Beyond the CSV outputs above, the returned ``EventLog`` object also supports:

.. code-block:: python

   event_log.to_xes("event_log.xes")     # XES for process-mining tools
   event_log.to_pm4py()                  # PM4Py-compatible event log object

See :doc:`usage` for how to obtain the ``EventLog`` object.

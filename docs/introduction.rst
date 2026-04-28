Introduction
============

Why Sensor2EventLog?
--------------------

Modern industrial processes (food production, chemical plants, manufacturing lines) generate continuous streams of 
sensor data — temperature, flow, pressure, valve positions. Process mining tools, however, expect *event logs*: discrete records 
of the form *(case, activity, timestamp)*. Bridging this gap is non-trivial because raw sensor signals do not carry semantic labels for process activities.

Sensor2EventLog tackles this by combining domain expertise with ML models like Hidden
Markov Model (HMM). Domain experts encode their knowledge as feature rules
and process-state hypotheses; the HMM then segments the sensor stream into
labeled intervals that form a clean event log suitable for downstream
process mining.

The Machine Teaching loop
-------------------------

Rather than treating model building as a one-shot training step, the framework
embeds it in an interactive *teaching loop*:

* **Plan** — the expert specifies which sensor features and which event
  predicates are likely to discriminate process states.
* **Explain** — the framework computes those features, fits an HMM, and
  reports diagnostics back to the expert (state distributions, confusion
  matrices, rule performance).
* **Review** — the expert inspects the results, refines the feature plan
  or rule set, and re-runs the loop.

This makes the modeling process transparent and auditable, which matters
in regulated domains such as food and pharmaceutical production.

Who is this for?
----------------

* Process engineers who have sensor histories and want to extract event logs.
* Process-mining researchers working with IoT or SCADA data sources.
* Practitioners exploring Machine Teaching as an alternative to pure end-to-end deep learning.

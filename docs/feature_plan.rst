Feature Plan Reference
======================

The feature plan is a Python dictionary that tells the abstraction layer
which features to compute. Each key corresponds to a feature family.

Schema
------

.. code-block:: python

   feature_plan = {
       'statistical': [<sensor>, ...],
       'temporal':    [<sensor>, ...],
       'stability':   [<sensor>, ...],
       'interaction': [[<sensor>, <sensor>, ...], ...],
       'event':       [<predicate string>, ...],
       'contextual':  [...],
   }

Statistical
-----------

Rolling-window descriptors of a single sensor channel. Window sizes are taken
from ``config.FEATURE_CONFIG["window_sizes"]``.

.. code-block:: python

   'statistical': ['T', 'Q_in', 'Q_out']

Temporal
--------

First-difference and smoothed-difference features for trend detection.

Stability
---------

Boolean flags indicating whether a sensor is in a stable plateau, controlled
by ``config.FEATURE_CONFIG["stability_eps"]``.

Interaction
-----------

Cross-sensor features computed over groups of channels:

.. code-block:: python

   'interaction': [['T', 'Q_in', 'Q_out']]

Event predicates
----------------

String predicates evaluated against the engineered feature columns. Both
``&`` / ``|`` and ``AND`` / ``OR`` are accepted as logical connectives.

.. code-block:: python

   'event': [
       '(T_diff_smooth > 1)',
       '(T > 70) & (T_stable_flag == 1)',
       '(Q_in > 0.3) AND (T_diff < 0.2)',
   ]

Each predicate becomes a binary column prefixed with ``event_`` that the HMM
consumes alongside the raw rolling means.

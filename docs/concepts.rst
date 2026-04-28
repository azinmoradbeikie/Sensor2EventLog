Core Concepts
=============

Modular abstraction layer
-------------------------

The abstraction layer sits between raw sensor signals and the HMM. It is
composed of independent feature families that the expert switches on or off
through the *feature plan*. Each family produces engineered columns the HMM
consumes as Gaussian observations.

The five built-in families are:

* **Statistical** — rolling means, standard deviations, min/max windows.
* **Temporal** — first differences, smoothed differences, time-since-event.
* **Stability** — flags marking sensor plateaus or drifts.
* **Interaction** — cross-sensor combinations (ratios, sums).
* **Event** — boolean predicates encoding expert rules
  (e.g. ``(T > 70) & (T_stable_flag == 1)``).

Supervised vs unsupervised HMM
------------------------------

The pipeline supports two modes:

* **Supervised** — initial transition, start, and emission parameters are
  seeded from labeled training data, then refined via Baum-Welch. Use this
  when you have ground-truth state labels for at least some batches.
* **Unsupervised** — a vanilla Gaussian HMM is fit and its discovered states
  are aligned to the expert's state list using the Hungarian algorithm.
  Use this when labels are scarce or absent.

Both modes decode test sequences with the Viterbi algorithm and produce the
same downstream artifacts.

From state sequence to event log
--------------------------------

The decoded per-timestamp state sequence is converted into intervals
(``case_id``, ``state``, ``start``, ``end``). Brief flickers shorter than a
configurable threshold (default 2 seconds) are filtered out to remove
spurious transitions. The resulting interval log is the event log used by
downstream process-mining tools.

<p align="center">
  <img src="https://raw.githubusercontent.com/azinmoradbeikie/Sensor2EventLog/main/images/Sensor2EventLog_001.png" width="300" />
</p>
<h1 align="center">Sensor2EventLog</h1>
<p align="center"><em>Knowledge-guided eventization of continuous sensor streams for process mining.</em></p>
<p align="center">
  <a href="https://sensor2eventlog.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/sensor2eventlog/badge/?version=latest" alt="Documentation Status"/></a>
  <a href="https://pypi.org/project/sensor2eventlog/"><img src="https://img.shields.io/pypi/v/sensor2eventlog" alt="PyPI"/></a>
  <a href="https://doi.org/10.1007/978-3-032-28110-4_10"><img src="https://img.shields.io/badge/DOI-10.1007%2F978--3--032--28110--4__10-blue" alt="DOI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
</p>

## Overview

**Sensor2EventLog** turns raw, continuous sensor data (temperature, flow, pressure, valve state, ...) into
process-aware **event logs** that process-mining tools can consume directly. It does this by combining domain
expertise with a Hidden Markov Model (HMM) inside an interactive **Machine Teaching** loop: an expert plans which
features and rules should matter, the framework explains what the model actually learned, and the expert reviews
and refines the plan — all before a single event is committed to the log.

## Why Sensor2EventLog?

Industrial and IoT processes (food production, chemical plants, manufacturing lines, pasteurizers) generate dense,
continuous sensor streams. Process mining, on the other hand, expects discrete, semantically labeled records of the
form *(case, activity, timestamp)*. Bridging that gap by hand is slow and hard to justify; bridging it with a
black-box model is fast but opaque. Sensor2EventLog is built for the middle ground: a transparent, auditable
pipeline where a domain expert's knowledge directly shapes how sensor behavior is translated into process
semantics — important in regulated domains such as food and pharmaceutical manufacturing.


## Key features

- **Modular abstraction layer** with five pluggable feature families: statistical, temporal, stability,
  interaction, and event (rule-based) predicates.
- **Machine Teaching loop** — Plan → Explain → Review — so experts iterate on feature/rule quality instead of
  tuning a black box.
- **Supervised and unsupervised HMM modes**: seed from labeled batches when you have them, or fall back to
  unsupervised state discovery (aligned via the Hungarian algorithm) when you don't.
- **Rule diagnostics** — coverage, precision, and explainability metrics per rule and per state, with concrete
  recommendations for improving a weak feature plan.
- **Event log export** to CSV, XES, and PM4Py-compatible objects, plus Gantt-chart visualization.

## Installation

```bash
git clone https://github.com/azinmoradbeikie/Sensor2EventLog.git
cd Sensor2EventLog
pip install -r requirements.txt
```

Or install the released package from PyPI:

```bash
pip install sensor2eventlog
```

## Quickstart

Run the full pipeline on the bundled pasteurization dataset:

```bash
python3 main.py
```

Or run the self-contained toy walkthrough (synthetic data, no external files needed):

```bash
python3 tutorial/toy_walkthrough.py
```

A minimal feature plan looks like this:

```python
feature_plan = {
    "statistical":  ["T", "Q_in", "Q_out"],
    "temporal":     ["T", "Q_in", "Q_out"],
    "stability":    ["T"],
    "interaction":  [["T", "Q_in"], ["T", "Q_out"]],
    "event": [
        {"fill_rule": "(Q_in > 0.5) & (Q_out < 0.1)"},
        {"hold_rule": "(T > 70) & (T_stable_flag == 1)"},
    ],
    "contextual":   ["batch_position"],
}
```

## Tutorials

| Level | Tutorial | Description |
|---|---|---|
| 0 | [`tutorial/level_0_concepts/`](tutorial/level_0_concepts.ipynb) | What an event log is, why raw sensor data isn't one, tiny raw-data-to-event-log example. |
| 1 | [`tutorial/`](tutorial/README.md) | Self-contained toy walkthrough — synthetic dataset, feature extraction, diagnostics, event log. Available now as script + notebook. |
| 1.5 | [`tutorials/level_1_5_first_event_log/`](tutorials/level_1_5_first_event_log/) | Build your first event log from a CSV with 2–3 rules; export CSV and XES. *(in progress)* |
| 2 | [`tutorials/level_2_machine_teaching_loop/`](tutorials/level_2_machine_teaching_loop/) | The full Plan → Explain → Review loop, before/after a weak feature plan. *(in progress)* |
| 2 | [`tutorials/level_2_hmm_supervised_vs_unsupervised/`](tutorials/level_2_hmm_supervised_vs_unsupervised/) | Same stream, both HMM modes, segmentation-quality comparison. *(in progress)* |
| 2 | [`tutorials/level_2_event_predicates/`](tutorials/level_2_event_predicates/) | The event-predicate DSL in depth. *(in progress)* |
| 3 | [`tutorials/level_3_real_pasteurization_case/`](tutorials/level_3_real_pasteurization_case/) | The bundled pasteurization dataset end-to-end, including where the abstraction fails. *(in progress)* |
| 3 | [`tutorials/level_3_eventization_quality/`](tutorials/level_3_eventization_quality/) | Evaluating generated events against expert annotations. *(in progress)* |
| 3 | [`tutorials/level_3_process_mining_integration/`](tutorials/level_3_process_mining_integration/) | Export to PM4Py, discover a DFG/Petri net/BPMN, run conformance analysis. *(in progress)* |

> The tutorial restructure is underway — Level 1 is available today, the rest are being migrated from internal
> notebooks. Follow [issues](https://github.com/azinmoradbeikie/Sensor2EventLog/issues) for progress.

## Machine Teaching loop

1. **Plan** — the expert specifies which sensor features and event predicates are likely to discriminate process
   states.
2. **Explain** — the framework computes those features, fits an HMM, and reports diagnostics back (state
   distributions, rule coverage/precision, confusion patterns).
3. **Review** — the expert inspects the diagnostics, refines the feature plan or rule set, and re-runs the loop.

This keeps modeling transparent and auditable instead of a one-shot black-box fit.

## Example: pasteurization process

The bundled example models a batch pasteurization process with states `Idle → Fill → HeatUp → Hold → Cool →
Discharge`, driven by temperature (`T`) and flow (`Q_in`, `Q_out`) sensors. Running `main.py` or the toy walkthrough
produces the segmented event log for this process along with a Gantt-chart visualization of batch timelines.

## Outputs

- `event_log.csv` — the raw interval event log (`case_id`, `activity`, `start`, `end`, `duration`).
- `filtered_log.csv` — the same log with brief, spurious segments removed.
- `gantt_chart.png` — a Gantt-style visualization of activities per case.
- `results.txt` — a text summary of the run, including diagnostics.

## Process mining integration

`EventLog` objects export directly to XES (`to_xes`) and to PM4Py-compatible structures (`to_pm4py`), so you can
feed Sensor2EventLog output straight into directly-follows graph discovery, Petri net/BPMN discovery, and
conformance or performance analysis in PM4Py. A dedicated walkthrough of this is planned in
[`tutorials/level_3_process_mining_integration/`](tutorials/level_3_process_mining_integration/).

## Eventization quality

The `evaluation` module's `RuleDiagnosticAnalyzer` scores each rule and state on coverage, precision, and
explainability against ground-truth labels, and surfaces prioritized recommendations for improving a feature plan.
A dedicated tutorial covering event-detection precision/recall, state-label accuracy, trace-level similarity, and
process-model quality is planned in
[`tutorials/level_3_eventization_quality/`](tutorials/level_3_eventization_quality/).

## Documentation

Full documentation is available on [Read the Docs](https://sensor2eventlog.readthedocs.io/en/latest/).

## Citation

If you use Sensor2EventLog in your research, please cite:

> Moradbeikie, A., Grigore, I. M., Lopes, S. I., & Barbon Junior, S. (2026). Sensor2EventLog: Bridging Continuous
> IoT Data and Process Mining through Eventization. In *International Conference on Advanced Information Systems
> Engineering* (pp. 177–194). Springer, Cham. https://doi.org/10.1007/978-3-032-28110-4_10

BibTeX and other citation formats are available in [`CITATION.cff`](CITATION.cff) — GitHub also renders a "Cite
this repository" button on the repo sidebar once this file is present.

## Authors

- Azin Moradbeikie
- I. M. Grigore
- S. I. Lopes
- S. Barbon Junior

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

<!-- TODO: add funding sources, testbeds (e.g. CiTin), or collaborating institutions here. -->

# causally-informed-prescriptive-monitoring

---

This structured approach provides a comprehensive framework for integrating causal inference into predictive process monitoring, demonstrating the tangible benefits through real-life case studies.

View the [paper](https://example.com/).

---

## Abstract

Predictive process monitoring (PPM) is focused on the prediction of different aspects of running process instances based on information of past executions. On the other hand, causal process mining aims to identify cause-and-effect relationships within business processes. In this paper, we analyze how causal inference can improve the performance of a predictive model using causal graph discovery. This perspective has been barely used for predictive monitoring. Our proposal has been experimentally tested using three real-life case studies. Positive results obtained confirm the benefits of the incorporation of causal in- formation in the predictive monitoring process in terms of accuracy and time consumption.

## Structure

- **data**: This directory contains all the event logs used in the experiments.
- **experiments**: This directory contains the necesary scripts and notebooks for running the experiments.
- **images**: This directory is for storing images generated during the experiments (e.g., causal graphs, plots).
- **models**: This directory is for storing trained models.
- **outputs**: This directory contains the outputs of the experiment.
- **requirements.txt**: List of the dependencies required to run the experiment.

```bash
.
├── data
│   ├── BPI_Challenge_2013_incident.csv
│   ├── bpi2015_1.csv
│   ├── bpi2015_2.csv
│   └── traffic_fines_1.csv
├── experiments
│   ├── causality.py
│   ├── experiment_bpi2013.ipynb
│   ├── experiment_bpi2015_1.ipynb
│   ├── experiment_bpi2015_2.ipynb
│   └── experiment_traffic.ipynb
├── images
├── models
├── outputs
├── README.md
└── requirements.txt
```

## Running the experiments

To replicate the experiments:

1. **Create a virtual environment**:

```bash
python -m venv venv
```

2. **Activate the virtual environment**:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS or Linux:

```bash
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the experiments**.

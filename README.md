# AutoMLapp

A Streamlit application for CSV exploration, baseline AutoML experiments, and practical NLP utilities.

## Features

- Upload a CSV dataset and generate an interactive exploratory-data-analysis dashboard.
- Create interactive Plotly scatter plots.
- Compare regression or classification baselines with PyCaret, or run DBSCAN clustering.
- Download the trained model.
- Use PDF question answering, sentiment analysis, zero-shot classification, and grammar correction.

## Project structure

```text
AutoMLapp/
├── app.py                         # Streamlit entry point
├── src/automl_app/
│   ├── app.py                     # UI pages and application flow
│   └── services/document_qa.py    # PDF retrieval-QA service
├── requirements.txt
├── .env.example
└── .gitignore
```

Generated files such as uploaded datasets, models and PDFs are excluded from Git.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/younesgu/AutoMLapp.git
cd AutoMLapp
py -3.11 -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch the application:

```bash
python -m pip install --upgrade pip
python -m pip install --prefer-binary -r requirements.txt
python -m streamlit run app.py
```

If you installed an earlier version of this project, recreate `.venv` before
installing. The old environment may contain incompatible versions of pandas and
YData Profiling. The application now uses a built-in analysis dashboard and no
longer depends on YData Profiling.

## PDF question answering

This feature downloads Hugging Face models and uses a hosted language model.
Create `.env`, then add your own token




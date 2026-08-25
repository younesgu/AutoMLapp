"""Streamlit user interface for AutoMLapp."""

import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from automl_app.services.document_qa import answer_document

DATASET_PATH = Path("dataset.csv")
MODEL_PATH = Path("best_model.pkl")
PDF_PATH = Path("uploaded_file.pdf")


def load_dataset() -> pd.DataFrame | None:
    return pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else None


def require_dataset(dataframe: pd.DataFrame | None) -> bool:
    if dataframe is None:
        st.info("Upload a CSV dataset first.")
        return False
    return True


def render_upload() -> None:
    st.title("Upload your dataset")
    file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if file:
        dataframe = pd.read_csv(file)
        dataframe.to_csv(DATASET_PATH, index=False)
        st.success("Dataset uploaded successfully.")
        st.dataframe(dataframe, use_container_width=True)


def render_analysis(dataframe: pd.DataFrame | None) -> None:
    st.title("Exploratory data analysis")
    if not require_dataset(dataframe):
        return

    row_count, column_count = dataframe.shape
    missing_count = int(dataframe.isna().sum().sum())
    duplicate_count = int(dataframe.duplicated().sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{row_count:,}")
    metric_columns[1].metric("Columns", f"{column_count:,}")
    metric_columns[2].metric("Missing values", f"{missing_count:,}")
    metric_columns[3].metric("Duplicate rows", f"{duplicate_count:,}")

    overview_tab, numeric_tab, categorical_tab, quality_tab = st.tabs(
        ["Overview", "Numeric columns", "Categorical columns", "Data quality"]
    )

    with overview_tab:
        st.subheader("Dataset preview")
        st.dataframe(dataframe.head(100), use_container_width=True)

        column_summary = pd.DataFrame(
            {
                "Type": dataframe.dtypes.astype(str),
                "Non-null": dataframe.notna().sum(),
                "Missing": dataframe.isna().sum(),
                "Missing %": (dataframe.isna().mean() * 100).round(2),
                "Unique": dataframe.nunique(dropna=True),
            }
        )
        st.subheader("Column summary")
        st.dataframe(column_summary, use_container_width=True)

        st.subheader("Descriptive statistics")
        try:
            statistics = dataframe.describe(include="all").transpose()
            st.dataframe(statistics, use_container_width=True)
        except ValueError:
            st.info("No descriptive statistics are available for this dataset.")

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    with numeric_tab:
        if not numeric_columns:
            st.info("No numeric column was detected.")
        else:
            selected_numeric = st.selectbox(
                "Numeric column", numeric_columns, key="analysis_numeric_column"
            )
            chart_columns = st.columns(2)
            chart_columns[0].plotly_chart(
                px.histogram(
                    dataframe,
                    x=selected_numeric,
                    nbins=30,
                    title=f"Distribution of {selected_numeric}",
                ),
                use_container_width=True,
            )
            chart_columns[1].plotly_chart(
                px.box(
                    dataframe,
                    y=selected_numeric,
                    points="outliers",
                    title=f"Box plot of {selected_numeric}",
                ),
                use_container_width=True,
            )

            if len(numeric_columns) >= 2:
                correlations = dataframe[numeric_columns].corr()
                correlation_chart = go.Figure(
                    data=go.Heatmap(
                        z=correlations.to_numpy(),
                        x=correlations.columns,
                        y=correlations.index,
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1,
                        colorbar={"title": "Correlation"},
                    )
                )
                correlation_chart.update_layout(
                    title="Correlation matrix",
                    xaxis_title="Feature",
                    yaxis_title="Feature",
                )
                st.plotly_chart(correlation_chart, use_container_width=True)

    categorical_columns = [
        column for column in dataframe.columns if column not in numeric_columns
    ]
    with categorical_tab:
        if not categorical_columns:
            st.info("No categorical column was detected.")
        else:
            selected_category = st.selectbox(
                "Categorical column",
                categorical_columns,
                key="analysis_categorical_column",
            )
            category_counts = (
                dataframe[selected_category]
                .astype("string")
                .fillna("<missing>")
                .value_counts()
                .head(20)
                .rename_axis("Value")
                .reset_index(name="Count")
            )
            st.plotly_chart(
                px.bar(
                    category_counts,
                    x="Count",
                    y="Value",
                    orientation="h",
                    title=f"Top values for {selected_category}",
                ),
                use_container_width=True,
            )

    with quality_tab:
        missing_by_column = dataframe.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values()
        if missing_by_column.empty:
            st.success("No missing values were detected.")
        else:
            missing_table = missing_by_column.rename("Missing").reset_index()
            missing_table.columns = ["Column", "Missing"]
            st.plotly_chart(
                px.bar(
                    missing_table,
                    x="Missing",
                    y="Column",
                    orientation="h",
                    title="Missing values by column",
                ),
                use_container_width=True,
            )

        if duplicate_count:
            st.warning(f"The dataset contains {duplicate_count:,} duplicate rows.")
        else:
            st.success("No duplicate rows were detected.")


def render_visualisation(dataframe: pd.DataFrame | None) -> None:
    st.title("Interactive visualisation")
    if not require_dataset(dataframe):
        return
    numeric = dataframe.select_dtypes(include="number").columns.tolist()
    if len(numeric) < 2:
        st.warning("The dataset needs at least two numeric columns for a scatter plot.")
        return
    x = st.selectbox("X axis", numeric)
    y = st.selectbox("Y axis", numeric, index=1)
    color = st.selectbox("Colour", dataframe.columns)
    st.plotly_chart(
        px.scatter(dataframe, x=x, y=y, color=color), use_container_width=True
    )


def render_models(dataframe: pd.DataFrame | None) -> None:
    st.title("Machine-learning models")

    if not require_dataset(dataframe):
        return

    target = st.selectbox("Target column", dataframe.columns)

    kind = st.selectbox(
        "Problem type",
        ["Regression", "Classification", "Clustering"],
    )

    if not st.button("Run modelling"):
        return

    with st.spinner("Training and comparing models..."):

        if kind == "Regression":
            from pycaret.regression import (
                compare_models,
                pull,
                save_model,
                setup,
            )

            setup(
                data=dataframe,
                target=target,
                session_id=42,
                verbose=False,
            )

            model = compare_models()
            scores = pull().copy()

        elif kind == "Classification":
            from pycaret.classification import (
                compare_models,
                pull,
                save_model,
                setup,
            )

            setup(
                data=dataframe,
                target=target,
                session_id=42,
                verbose=False,
            )

            model = compare_models()
            scores = pull().copy()

        else:
            from pycaret.clustering import (
                create_model,
                pull,
                save_model,
                setup,
            )

            setup(
                data=dataframe,
                session_id=42,
                verbose=False,
            )

            model = create_model("dbscan")
            scores = pull().copy()

        save_model(
            model,
            str(MODEL_PATH.with_suffix("")),
        )

    st.success("Modelling complete.")

    st.subheader("Selected model")
    st.info(type(model).__name__)
    st.code(str(model), language="text")

    st.subheader("Model scores")
    st.dataframe(
        scores,
        use_container_width=True,
        hide_index=True,
    )

    if hasattr(model, "get_params"):
        with st.expander("Model parameters"):
            parameters = pd.DataFrame(
                {
                    "Parameter": model.get_params().keys(),
                    "Value": [
                        str(value)
                        for value in model.get_params().values()
                    ],
                }
            )

            st.dataframe(
                parameters,
                use_container_width=True,
                hide_index=True,
            )

    st.caption(f"Model saved to: {MODEL_PATH}")


def render_nlp() -> None:
    st.title("Natural-language processing")
    task = st.selectbox(
        "Task",
        ["Document Q&A", "Sentiment analysis", "Text classification", "Spell check"],
    )
    if task == "Document Q&A":
        pdf = st.file_uploader("Upload a PDF", type=["pdf"])
        if pdf:
            contents = pdf.getvalue()
            PDF_PATH.write_bytes(contents)
            encoded = base64.b64encode(contents).decode("utf-8")
            st.markdown(
                f'<embed src="data:application/pdf;base64,{encoded}" width="700" height="600">',
                unsafe_allow_html=True,
            )
        question = st.text_area("Question", height=100)
        if st.button("Answer") and question:
            if not PDF_PATH.exists():
                st.warning("Upload a PDF first.")
            else:
                try:
                    st.write(answer_document(str(PDF_PATH), question))
                except RuntimeError as error:
                    st.error(str(error))
    elif task == "Sentiment analysis":
        text = st.text_area("Sentence", height=100)
        if st.button("Predict sentiment") and text:
            from transformers import pipeline

            result = pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa")(
                text
            )[0]
            st.write(f"Sentiment: {result['label']} ({result['score']:.2%})")
    elif task == "Text classification":
        text = st.text_input("Sentence")
        labels = st.text_input("Candidate labels (comma-separated)")
        if st.button("Classify") and text and labels:
            from transformers import pipeline

            values = [item.strip() for item in labels.split(",") if item.strip()]
            result = pipeline(
                "zero-shot-classification", model="facebook/bart-large-mnli"
            )(text, values)
            st.write(
                f"Classification: {result['labels'][0]} ({result['scores'][0]:.2%})"
            )
    else:
        text = st.text_area("Sentence to correct", height=100)
        if st.button("Correct") and text:
            from happytransformer import HappyTextToText, TTSettings

            model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
            st.write(
                model.generate_text(
                    text, args=TTSettings(num_beams=5, min_length=1)
                ).text
            )


def main() -> None:
    st.set_page_config(page_title="AutoMLapp", page_icon="🤖", layout="wide")
    st.sidebar.title("AutoMLapp")
    st.sidebar.caption("Automated ML and practical NLP workflows.")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Upload dataset",
            "Data analysis",
            "Data visualisation",
            "ML models",
            "Download model",
            "NLP",
        ],
    )
    dataframe = load_dataset()
    if page == "Upload dataset":
        render_upload()
    elif page == "Data analysis":
        render_analysis(dataframe)
    elif page == "Data visualisation":
        render_visualisation(dataframe)
    elif page == "ML models":
        render_models(dataframe)
    elif page == "Download model":
        st.title("Download model")
        if MODEL_PATH.exists():
            st.download_button(
                "Download best model",
                MODEL_PATH.read_bytes(),
                file_name=MODEL_PATH.name,
            )
        else:
            st.info("Run a modelling experiment first.")
    else:
        render_nlp()

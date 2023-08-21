import pandas as pd
import streamlit as st
import pygwalker as pyg
from sklearn.cluster import KMeans
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Data Analysis: Demo",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("Data Analysis: Demo")

st.header("Clustering")

stripping_df = None

csv_data = st.file_uploader("Upload Raw data file for stripping", type="csv")

if csv_data is not None:
    stripping_df = pd.read_csv(csv_data, sep=";")
    # cols = stripping_df.select_dtypes(include=["float64"]).columns
    stripping_df[
        ["Abisolierungs-Einzeldefektflaeche_max",
            "Abisolierungs-Gesamtdefektflaeche"]
    ] = stripping_df[
        ["Abisolierungs-Einzeldefektflaeche_max",
            "Abisolierungs-Gesamtdefektflaeche"]
    ].round(
        2
    )

dataframe_tab, graph_tab = st.tabs(["Dataframe", "Graph"])

with dataframe_tab:
    if stripping_df is not None:
        table = st.dataframe(stripping_df)

cluster_button = st.button(
    "Cluster",
    key="cluster_button",
    use_container_width=True,
    type="primary",
)

if cluster_button and stripping_df is not None:
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(
        stripping_df[
            [
                "Abisolierungs-Einzeldefektflaeche_max",
                "Abisolierungs-Gesamtdefektflaeche",
            ]
        ].fillna(0)
    )
    stripping_df["Label"] = kmeans.labels_
    table.dataframe(stripping_df)

with graph_tab:
    # Generate the HTML using Pygwalker
    pyg_html = pyg.walk(stripping_df, return_html=True)
    components.html(pyg_html, height=1000, scrolling=True)

corr_button = st.button(
    "Correlation Analysis",
    key="corr_button",
    use_container_width=True,
    type="primary",
)

if corr_button and stripping_df is not None:
    corr_df = stripping_df.corr()
    st.write(corr_df)

database_upload_button = st.button(
    "Upload the data to Neo4j",
    key="upload_button",
    use_container_width=True,
    type="primary",
)

if database_upload_button:
    st.info("This is where the data would be uploaded to Neo4j")

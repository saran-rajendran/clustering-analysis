import glob
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

csv_data = st.file_uploader(
    "Upload Raw data file for stripping", type="csv", accept_multiple_files=True)

stripping_df = pd.DataFrame()

if csv_data is not None:
    if isinstance(csv_data, list):
        for file in csv_data:
            mach_no = file.name.split('_')[-2]
            x = pd.read_csv(file, sep=";")
            x['Machine_Number'] = mach_no
            if (not stripping_df.empty) and all(x.columns == stripping_df.columns):
                stripping_df = pd.concat([stripping_df, x], axis=0)
            elif stripping_df.empty:
                stripping_df = pd.concat([stripping_df, x], axis=0)
    else:
        stripping_df = pd.read_csv(csv_data, sep=";")

    stripping_df = stripping_df.reset_index()
    # print(stripping_df.columns)
    # cols = stripping_df.select_dtypes(include=["float64"]).columns
    stripping_df[
        ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max", "Abisolierungs-Einzeldefektflaeche_max",
            "Abisolierungs-Gesamtdefektflaeche"]
    ] = stripping_df[
        ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max", "Abisolierungs-Einzeldefektflaeche_max",
            "Abisolierungs-Gesamtdefektflaeche"]
    ].round(
        2
    )

dataframe_tab, graph_tab, cluster_tab = st.tabs(
    ["Dataframe", "Graph", "Clustering"])

with dataframe_tab:
    if stripping_df is not None:
        table = st.dataframe(stripping_df)
        st.write(stripping_df[['Machine_Number', 'Arbeitsfolge', 'Min_Differenz_Abisolierposition', 'Max_Differenz_Abisolierposition',
                               'Min_Differenz_Abisolierlaenge_max', 'Max_Differenz_Abisolierlaenge_max',
                               'Min_Abisolierungs-Einzeldefektflaeche_max', 'Max_Abisolierungs-Einzeldefektflaeche_max',
                               'Min_Abisolierungs-Gesamtdefektflaeche', 'Max_Abisolierungs-Gesamtdefektflaeche']].drop_duplicates())

# cluster_button = st.button(
#     "Cluster",
#     key="cluster_button",
#     use_container_width=True,
#     type="primary",
# )

# if cluster_button and stripping_df is not None:
#     kmeans = KMeans(n_clusters=5, random_state=0)
#     kmeans.fit(
#         stripping_df[
#             ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max",
#                 "Abisolierungs-Einzeldefektflaeche_max",
#                 "Abisolierungs-Gesamtdefektflaeche",
#              ]
#         ].fillna(0)
#     )
#     stripping_df["Label"] = kmeans.labels_
#     table.dataframe(stripping_df)

with graph_tab:
    # Generate the HTML using Pygwalker
    pyg_html = pyg.walk(stripping_df, return_html=True)
    components.html(pyg_html, height=1000, scrolling=True)

with cluster_tab:
    table_cluster = st.dataframe(stripping_df)
    k = st.number_input('Number of Clusters', value=5, key='k')
    cluster_button = st.button(
        "Cluster",
        key="cluster_button",
        use_container_width=True,
        type="primary",
    )
    if cluster_button and stripping_df is not None:
        cols = ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max",
                "Abisolierungs-Einzeldefektflaeche_max",
                "Abisolierungs-Gesamtdefektflaeche",
                ]
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(
            stripping_df[cols].fillna(0)
        )
        stripping_df["Label"] = kmeans.labels_
        counter = {}
        for lb in stripping_df["Label"].unique():
            lb_df = stripping_df[stripping_df["Label"] == lb]
            counter[lb] = lb_df['Ergebnis'].value_counts().to_dict()
        counter_df = pd.DataFrame(counter).fillna(0).T.sort_index()
        cluster_centers = kmeans.cluster_centers_
        cluster_qlty = {}
        labels = ['Critical', 'Very Bad', "Bad", "Good", "Very Good"]
        for row in counter_df.iterrows():
            cluster_qlty[row[0]] = (row[1]['Pass'] + 1) / \
                (row[1].sum() - row[1]['Pass'])
        cluster_qlty_df = pd.DataFrame({f"{l}_{k[0]}": k[1] for l, k in zip(
            labels, sorted(cluster_qlty.items(), key=lambda item: item[1]))}, index=[0])
        table_cluster.dataframe(stripping_df)
        st.write(f"Features used for Clustering: {cols}")
        st.write("Cluster Centers: ")
        st.write(pd.DataFrame(cluster_centers))
        st.write("Result Counter: ")
        st.write(counter_df)
        st.write("Cluster Labels: ")
        st.write(cluster_qlty_df)

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

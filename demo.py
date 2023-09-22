import numpy as np
import pandas as pd
import streamlit as st
import pygwalker as pyg
import plotly.express as px
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

    stripping_df = stripping_df.reset_index(drop=True)

    stripping_df[
        ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max", "Abisolierungs-Einzeldefektflaeche_max",
            "Abisolierungs-Gesamtdefektflaeche"]
    ] = stripping_df[
        ["Differenz_Abisolierposition", "Differenz_Abisolierlaenge_max", "Abisolierungs-Einzeldefektflaeche_max",
            "Abisolierungs-Gesamtdefektflaeche"]
    ].round(
        2
    )


dataframe_tab, graph_tab, cluster_tab, visualize_tab = st.tabs(
    ["Dataframe", "Graph", "Clustering", "Visualization"])

with dataframe_tab:
    if stripping_df is not None:
        table = st.dataframe(stripping_df)
        st.write(stripping_df[['Machine_Number', 'Arbeitsfolge', 'Min_Differenz_Abisolierposition', 'Max_Differenz_Abisolierposition',
                               'Min_Differenz_Abisolierlaenge_max', 'Max_Differenz_Abisolierlaenge_max',
                               'Min_Abisolierungs-Einzeldefektflaeche_max', 'Max_Abisolierungs-Einzeldefektflaeche_max',
                               'Min_Abisolierungs-Gesamtdefektflaeche', 'Max_Abisolierungs-Gesamtdefektflaeche']].drop_duplicates())

with graph_tab:
    # Generate the HTML using Pygwalker
    pyg_html = pyg.walk(stripping_df, return_html=True)
    components.html(pyg_html, height=1000, scrolling=True)

with cluster_tab:
    table_cluster = st.dataframe(stripping_df)
    k = st.number_input('Number of Clusters', value=5, key='k')
    context = st.selectbox('Normalization Context', ('Strict', 'Wide'))
    st.write('You selected:', context)

    cols = [('Differenz_Abisolierposition', 'Min_Differenz_Abisolierposition', 'Max_Differenz_Abisolierposition'),
            ('Differenz_Abisolierlaenge_max', 'Min_Differenz_Abisolierlaenge_max',
             'Max_Differenz_Abisolierlaenge_max'),
            ('Abisolierungs-Einzeldefektflaeche_max', 'Min_Abisolierungs-Einzeldefektflaeche_max',
             'Max_Abisolierungs-Einzeldefektflaeche_max'),
            ('Abisolierungs-Gesamtdefektflaeche', 'Min_Abisolierungs-Gesamtdefektflaeche', 'Max_Abisolierungs-Gesamtdefektflaeche')]

    cluster_button = st.button(
        "Cluster",
        key="cluster_button",
        use_container_width=True,
        type="primary",
    )
    if cluster_button and stripping_df is not None:
        if context == 'Strict':
            for i, (a, b, c) in enumerate(cols):
                low_val = stripping_df.iloc[stripping_df.apply(
                    lambda x: abs(x[b] - x[c]), axis=1).idxmin()][b]
                up_val = stripping_df.iloc[stripping_df.apply(
                    lambda x: abs(x[b] - x[c]), axis=1).idxmin()][c]
                stripping_df[f"{b}_new"] = low_val
                stripping_df[f"{c}_new"] = up_val
                stripping_df[f"Target_{a}"] = 0.5 if i < 2 else 0
                stripping_df[f"{a}_norm"] = (
                    stripping_df[a] - low_val) / (up_val - low_val)
        elif context == 'Wide':
            for i, (a, b, c) in enumerate(cols):
                low_val = stripping_df.iloc[stripping_df.apply(
                    lambda x: x[b] - x[c], axis=1).idxmin()][b]
                up_val = stripping_df.iloc[stripping_df.apply(
                    lambda x: x[b] - x[c], axis=1).idxmax()][c]
                stripping_df[f"{b}_new"] = low_val
                stripping_df[f"{c}_new"] = up_val
                stripping_df[f"Target_{a}"] = 0.5 if i < 2 else 0
                stripping_df[f"{a}_norm"] = (
                    stripping_df[a] - low_val) / (up_val - low_val)

        cols = ["Differenz_Abisolierposition_norm", "Differenz_Abisolierlaenge_max_norm",
                "Abisolierungs-Einzeldefektflaeche_max_norm",
                "Abisolierungs-Gesamtdefektflaeche_norm",
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
                (row[1].sum() - row[1]['Pass'] + 1)
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

with visualize_tab:
    # Create data
    features = ['Differenz_Abisolierposition_norm', 'Differenz_Abisolierlaenge_max_norm',
                'Abisolierungs-Einzeldefektflaeche_max_norm', 'Abisolierungs-Gesamtdefektflaeche']
    num_clusters = k

    # Create custom values
    thresholds = {
        "Differenz_Abisolierposition_norm": {"min": 0, "max": 1, "target": 0.5},
        "Differenz_Abisolierlaenge_max_norm": {"min": 0, "max": 1, "target": 0.5},
        "Abisolierungs-Einzeldefektflaeche_max_norm": {"min": 0, "max": 1, "target": 0},
        "Abisolierungs-Gesamtdefektflaeche": {"min": 0, "max": 1, "target": 0},
    }

    # Create data for box values
    box_data = {
        "Features": [],
        "Value": [],
        "Type": []
    }

    for feature in features:
        box_data["Features"].append(feature)
        box_data["Value"].append(thresholds[feature]["min"])
        box_data["Type"].append("Min")

        box_data["Features"].append(feature)
        box_data["Value"].append(thresholds[feature]["max"])
        box_data["Type"].append("Max")

        box_data["Features"].append(feature)
        box_data["Value"].append(thresholds[feature]["target"])
        box_data["Type"].append("Target")

    df_box = pd.DataFrame(box_data)

    # Create data for cluster centers
    cluster_data = {
        "Features": [],
        "Value": [],
        "Cluster": [],
    }
    colors = {'Min': 'red', 'Max': 'red', 'Target': 'green'}

    for i, feature in enumerate(features):
        for cluster_num in range(num_clusters):
            cluster_data["Features"].append(feature)
            cluster_data["Value"].append(cluster_centers[cluster_num][i])
            cluster_data["Cluster"].append(f"Cluster {cluster_num}")

    df_cluster = pd.DataFrame(cluster_data)

    # Create figure
    fig = px.scatter(df_cluster, x="Features", y="Value",
                     color="Cluster", title="Customized Box Plots")
    fig.update_traces(marker=dict(size=8))

    # Add box traces for min, max, mean
    box_types = ["Min", "Max", "Target"]

    for box_type in box_types:
        df_filtered = df_box[df_box["Type"] == box_type]
        fig.add_trace(px.box(df_filtered, x="Features",
                             y="Value", color="Type", color_discrete_map=colors, points=False).data[0])

    fig.update_xaxes(categoryorder="array", categoryarray=features)
    st.plotly_chart(
        fig,
        use_container_width=True,
        sharing="streamlit",
        theme="streamlit",
    )

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

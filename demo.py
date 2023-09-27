import numpy as np
import pandas as pd
import streamlit as st
import pygwalker as pyg
import plotly.express as px
from sklearn.cluster import KMeans
import streamlit.components.v1 as components

import config
from neo import upload_to_neo4j

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

dataframe_tab, graph_tab, cluster_tab, visualize_tab, upload_tab = st.tabs(
    ["Dataframe", "Graph", "Clustering", "Visualization", "Upload"])

with dataframe_tab:
    if not stripping_df.empty:
        features = list(config.features_with_threshold.keys())
        selected_features = {
            feature: True if feature in config.default_features else False for feature in features}

        select_all = st.checkbox("Select All")
        if select_all:
            selected_features = {option: True for option in features}

        selected_features.update({feature: st.checkbox(
            feature, value=selected_features[feature]) for feature in features})

        select_feats = [k for k, v in selected_features.items() if v is True]
        stripping_df[
            select_feats
        ] = stripping_df[
            select_feats
        ].round(2)
        table = st.dataframe(stripping_df)

        cols = ['Machine_Number', 'Arbeitsfolge']
        for feat in select_feats:
            cols.extend(config.features_with_threshold[feat][1:])
        st.write(stripping_df[cols].drop_duplicates())

with graph_tab:
    pyg_html = pyg.walk(stripping_df, return_html=True)
    components.html(pyg_html, height=1000, scrolling=True)

with cluster_tab:
    table_cluster = st.dataframe(stripping_df)
    k = st.number_input('Number of Clusters', value=5, key='k')
    context = st.selectbox('Normalization Context', ('Strict', 'Wide'))
    st.write('You selected:', context)

    cols = []
    for feat in select_feats:
        cols.append(tuple(config.features_with_threshold[feat]))

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

        cols = []
        for feat in select_feats:
            cols.append(f'{feat}_norm')

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

        meta_data = {}
        thresholds = config.thresholds
        for i, cluster in enumerate(cluster_centers):
            temp = {}
            for j, feature_center in enumerate(cluster):
                temp[cols[j]] = {}
                temp[cols[j]]['within limit'] = thresholds[cols[j]
                                                           ]['min'] <= feature_center <= thresholds[cols[j]]['max']
                temp[cols[j]]['distance from target'] = abs(
                    thresholds[cols[j]]['target'] - feature_center)
                temp[cols[j]]['distance from min threshold'] = abs(
                    thresholds[cols[j]]['min'] - feature_center)
                temp[cols[j]]['distance from max threshold'] = abs(
                    thresholds[cols[j]]['max'] - feature_center)
            meta_data[i] = temp
        meta_data_df = pd.DataFrame.from_dict({(i, j): meta_data[i][j] for i in meta_data.keys(
        ) for j in meta_data[i].keys()}).T.reset_index().rename(columns={'level_0': 'Cluster', 'level_1': 'Feature'}).set_index('Cluster')

        table_cluster.dataframe(stripping_df)

        st.write(f"Features used for Clustering: {cols}")
        st.write("Cluster Centers: ")
        st.write(pd.DataFrame(cluster_centers))
        st.write("Result Counter: ")
        st.write(counter_df)
        st.write("Cluster Labels: ")
        st.write(cluster_qlty_df)

with visualize_tab:
    features = cols
    num_clusters = k

    box_data = {
        "Features": [],
        "Value": [],
        "Type": []
    }

    for feature in select_feats:
        feature = f'{feature}_norm'
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

    cluster_data = {
        "Features": [],
        "Value": [],
        "Cluster": [],
    }
    colors = {'Min': 'red', 'Max': 'red', 'Target': 'green'}

    for i, feature in enumerate(select_feats):
        feature = f'{feature}_norm'
        for cluster_num in range(num_clusters):
            cluster_data["Features"].append(feature)
            cluster_data["Value"].append(cluster_centers[cluster_num][i])
            cluster_data["Cluster"].append(f"Cluster {cluster_num}")

    df_cluster = pd.DataFrame(cluster_data)

    fig = px.strip(df_cluster, x="Features", y="Value",
                   color="Cluster", title="Customized Box Plots", stripmode='overlay')
    fig.update_traces(marker=dict(size=8))

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
    st.write('Meta Data: ')
    st.write(meta_data_df)

with upload_tab:
    if cluster_button and stripping_df is not None:
        upload_to_neo4j(stripping_df.iloc[:10, :])
        st.write('Data Uploaded to Neo4j database successfully')

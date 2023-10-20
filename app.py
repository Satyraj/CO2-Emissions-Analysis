# importing all functions

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pycountry

import helper

# inititalizing dataframe
df = pd.read_csv('Agrofood_co2_emission.csv')

# sidebar components
st.sidebar.title("Crops & Carbon")
# st.sidebar.image('iot-in-agriculture.jpg')
page = st.sidebar.radio("Navigation Menu", ["About the project", "Introduction", "Dataset", "Visualizations", "Clustering & Insights"])

if page == "About the project":
    st.sidebar.markdown(helper.about_me, unsafe_allow_html=True)
    st.markdown(helper.title_code, unsafe_allow_html=True)

elif page == "Introduction":
    st.image('Emissions-by-sector-pie-chart.png')
    st.title("Introduction")
    st.markdown(helper.introduction, unsafe_allow_html=True)

elif page == "Dataset":
    st.sidebar.markdown(helper.dataset_note, unsafe_allow_html=True)

    st.sidebar.header("Select Country and Year")

    selected_country = st.sidebar.selectbox("Select Country", ["All Countries"] + list(df["Area"].unique()))
    selected_year = st.sidebar.selectbox("Select Year", ["All Years"] + list(df["Year"].unique()))

    if selected_country == "All Countries" and selected_year == "All Years":
        filtered_df = df
    elif selected_country == "All Countries":
        filtered_df = df[df["Year"] == selected_year]
    elif selected_year == "All Years":
        filtered_df = df[df["Area"] == selected_country]
    else:
        filtered_df = df[(df["Area"] == selected_country) & (df["Year"] == selected_year)]

    st.header("Agricultural CO2 Emissions Dataset")
    st.write(filtered_df)
    st.markdown(helper.dataset_features, unsafe_allow_html=True)

elif page == "Visualizations":
    selected_vis = st.sidebar.selectbox("Select an option", ["Emissions per Year", "CO2 and Temperature", "Countrywise Emission Trends", "Yearwise Temperature Distribution", "Global Analysis", "Insights"])

    if selected_vis == "Emissions per Year":
        st.header("Emissions by year")
        st.write("What are the total agricultural emissions across the globe? Find out here.")
        selected_year = st.selectbox("Select a year:", df["Year"].unique())
        calc_gt = helper.calculate_co2_emissions(df, selected_year)
        st.success(f"The amount of CO2 from agrifood in {selected_year} is {calc_gt} gigatons (gt).")

    elif selected_vis == "CO2 and Temperature":
        st.header("Link between CO2 emissions & temperature")
        norm_temp = helper.co2_and_temp(df)
        fig, ax = plt.subplots(figsize=(12, 4))
        norm_temp.plot(ax=ax)
        plt.title("CO2 Emission & Temperature")
        st.pyplot(fig)

    elif selected_vis == "Countrywise Emission Trends":
        st.header("Countrywise CO2 Emission Trends")
        selected_nation = st.selectbox("Select a nation:", df["Area"].unique())
        plt = helper.plot_co2_trend(df, selected_nation)
        st.pyplot(plt)
    
    elif selected_vis == "Yearwise Temperature Distribution":
        st.header("Temperature distribution over the years")
        temperature_box_plot = helper.temperature_box_plot(df)
        st.plotly_chart(temperature_box_plot)

    elif selected_vis == "Global Analysis":
        st.header("Global Analysis")
        
        year = "dummy_value"
        # year = st.slider("Select a year:", min_value=1990, max_value=2020, step=1, value=2020)

        tab1, tab2, tab3 = st.tabs(["Temperature", "Emissions", "Per Capita Emissions"])

        fig1 = helper.global_temp_inc_map(helper.global_temp_inc(df,year))
        tab1.plotly_chart(fig1)

        fig2 = helper.global_emissions_map(helper.global_emissions(df,year))
        tab2.plotly_chart(fig2)

        tab2.header("Countrywise Emission Leaderboard")
        fig3 = helper.country_emission_chart(df, year)
        tab2.pyplot(fig3)

        fig4 = helper.per_capita_emissions_map(helper.per_capita_emissions(df,year))
        tab3.plotly_chart(fig4)

        tab3.header("Per Capita Emissions in Individual Countries")
        fig5 = helper.percapita_emission_chart(df)
        tab3.pyplot(fig5)

    elif selected_vis == "Insights":
        st.markdown(helper.insights, unsafe_allow_html=True)
        insight_year = st.slider("Select a year:", min_value=1990, max_value=2020, step=1, value=2020)
        fig = helper.continental_emission(df, year=insight_year)
        st.pyplot(fig) 

elif page == "Clustering & Insights":
    st.header("Clustering countries and deriving new insights")

    # clus = st.radio("Number of Clusters", [4,5,6,7,8])
    clus2 = st.slider("Number of Clusters (experimental): ", min_value=2, max_value=8, step=1, value=6)
    st.write("Baseline: 4 clusters, Optimal: 6 clusters")

    if st.button("Perform Clustering"):
        clustered_data = helper.perform_clustering(df, clus2)
        # st.write(clustered_data)

        map = helper.cluster_map(clustered_data)
        st.plotly_chart(map)

        scatterplot_fig = helper.cluster_scatterplot(clustered_data, 'Cluster')
        st.pyplot(scatterplot_fig) 

        st.markdown(helper.clustering_insights)  

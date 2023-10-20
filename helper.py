# importing all functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pycountry
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

title_code = """
<span style="color:green; font-size:80px; font-weight:600">Agriculture </span>
<span style="font-size:80px; font-weight:600"> CO2 </span>
<span style="color:blue; font-size:80px; font-weight:600">Emissions</span><br>

# This project includes...

### Exploratory Data Analysis (EDA):

- Investigating the relationship between CO2 emissions in the agri-food sector and temperature increase.
- Analyzing the impact of different countries in terms of aggregated data on emissions and temperature change.
- Identifying countries with the highest per capita emissions and their contribution to the overall impact.

### Data Preprocessing:

- Cleaning and preparing the data for machine learning model.
- Handling missing values & outliers, scaling and normalizing the data.

### Machine Learning Modeling:

- Developing a clustering model to explain the nature of various nations.
- Flexible clustering model with variable n_clusters
- Utilizing regression techniques to understand the relationship between CO2 emissions and temperature (*future plans*).
- Training and evaluating models to assess their performance (*future plans*).

### Interpretation and Insights:

- Analyzing the results of the visualizations and maps to gain insights into the impact of CO2 emissions on temperature changes.
- Drawing conclusions on the relationship between the agri-food sector and climate change.
- Providing recommendations for mitigating the impact of emissions in the agricultural sector.
"""

# about_me = """
# <span style="font-size:26px; font-weight:600">
#     Name: Jadeja Satyarajsinh <br>
#     Roll No.: 16<br>
#     Batch: C1<br>
#     Branch: IT
# </span>
# """
about_me = """
<span style="font-size:26px; font-weight:600">
<h1>About Us<br></h1>
    Name: Coding Bazz<br>
    1) Jadeja Satyarajsinh<br>
    2) Kalawadia Sachin<br>
    3) Naman Shah<br>
    4) Riddhi Mathasuriya
</span>
"""

introduction = """
**Global greenhouse gas emissions** come from a variety of **sectors**. Here, we see that the **energy sector** that comprises of **industrial energy use**, **transport** and **energy use in buildings** is the largest contributor in those emissions.

Then comes the **Agricultural Sector** which contributes in **18%** of global emissions. The idea of my project revolves around the same. But why focus on **Emissions due to Agriculture**?

Well, at the first glance, we cannot comprehend the fact that sector like **Agriculture** is a part of **greenhouse gas emissions**. But **analyzing** it in depth gives us **insights** as to how exactly are **agricultural activities** promoting emissions, particularly **carbon emissions** on which the dataset focuses.
"""
# <li><strong>Source: kaggle.com</strong></li>
dataset_note = """
<h3>About the dataset:</h3>
<ul>

<li>The agricultural CO2 emission dataset is created by merging and reprocessing around twelve individual datasets from FAO and IPCC.</li>
<li>The dataset focuses on CO2 emissions related to the agricultural sector, comprising about 18% of global annual emissions.</li>
</ul>
"""

dataset_features = """
<h3>Dataset Features:</h3>
<ul>
<li><strong>Savanna fires</strong>: Emissions from fires in savanna ecosystems.</li>
<li><strong>Forest fires</strong>: Emissions from fires in forested areas.</li>
<li><strong>Crop Residues</strong>: Emissions from burning or decomposing leftover plant material after crop harvesting.</li>
<li><strong>Rice Cultivation</strong>: Emissions from methane released during rice cultivation.</li>
<li><strong>Drained organic soils (CO2)</strong>: Emissions from carbon dioxide released when draining organic soils.</li>
<li><strong>Pesticides Manufacturing</strong>: Emissions from the production of pesticides.</li>
<li><strong>Food Transport</strong>: Emissions from transporting food products.</li>
<li><strong>Forestland</strong>: Land covered by forests.</li>
<li><strong>Net Forest conversion</strong>: Change in forest area due to deforestation and afforestation.</li>
<li><strong>Food Household Consumption</strong>: Emissions from food consumption at the household level.</li>
<li><strong>Food Retail</strong>: Emissions from the operation of retail establishments selling food.</li>
<li><strong>On-farm Electricity Use</strong>: Electricity consumption on farms.</li>
<li><strong>Food Packaging</strong>: Emissions from the production and disposal of food packaging materials.</li>
<li><strong>Agrifood Systems Waste Disposal</strong>: Emissions from waste disposal in the agrifood system.</li>
<li><strong>Food Processing</strong>: Emissions from processing food products.</li>
<li><strong>Fertilizers Manufacturing</strong>: Emissions from the production of fertilizers.</li>
<li><strong>IPPU</strong>: Emissions from industrial processes and product use.</li>
<li><strong>Manure applied to Soils</strong>: Emissions from applying animal manure to agricultural soils.</li>
<li><strong>Manure left on Pasture</strong>: Emissions from animal manure on pasture or grazing land.</li>
<li><strong>Manure Management</strong>: Emissions from managing and treating animal manure.</li>
<li><strong>Fires in organic soils</strong>: Emissions from fires in organic soils.</li>
<li><strong>Fires in humid tropical forests</strong>: Emissions from fires in humid tropical forests.</li>
<li><strong>On-farm energy use</strong>: Energy consumption on farms.</li>
<li><strong>Rural population</strong>: Number of people living in rural areas.</li>
<li><strong>Urban population</strong>: Number of people living in urban areas.</li>
<li><strong>Total Population - Male</strong>: Total number of male individuals in the population.</li>
<li><strong>Total Population - Female</strong>: Total number of female individuals in the population.</li>
<li><strong>total_emission</strong>: Total greenhouse gas emissions from various sources.</li>
<li><strong>Average Temperature °C</strong>: The average increasing of temperature (by year) in degrees Celsius,</li>
</ul>
"""

insights = """
# Insights:
As we see in this exploratory data analysis:
- **Asia** is the continent with the **highest CO2 emissions**.
- However, Asia's huge emission is strongly correlated with its large population. In fact, **Australia, followed by the United States and Canada, have the highest per capita emissions**, contrary to the graphical representation which placed countries like **Botswana and Bolivia** on top.
- Regarding **temperature**, specifically the average annual increase in Celsius, **Europe appears to be the continent most affected by climate change**, as observed in the graph below. However, in the global analysis, **Russia witnessed larger temperature rises**.
"""

def calculate_co2_emissions(df, selected_year):
    y_selected = df.loc[df["Year"] == selected_year]

    calc_kt = y_selected["total_emission"].sum() # Total emissions in kilotons
    calc_gt = round(calc_kt / 1000000, 2)  # Convert to gigatons

    return calc_gt

# Method: min-max scaling
def min_max_normalizer(df):
    norm = (df - df.min()) / (df.max() - df.min())
    return norm

def co2_and_temp(df):
    temp_emission = df.groupby("Year").agg({"Average Temperature °C": "mean", "total_emission": "mean", "Urban population": "mean"})
    norm_temp = min_max_normalizer(temp_emission)
    return norm_temp
    
def plot_co2_trend(df, nation):
    nation_df = df[df["Area"] == nation]
    nation_df = nation_df.set_index("Year")

    plt.figure(figsize=(12, 6))
    nation_df["total_emission"].plot(kind="line", color="green")
    plt.title(f"{nation}'s CO2 trend")
    plt.xlabel("Year")
    plt.ylabel("Total CO2 Emissions")
    return plt

def temperature_box_plot(df):
    fig = px.box(df, x="Year",
                 y="Average Temperature °C",
                 color="Year",
                 color_discrete_sequence=px.colors.sequential.Viridis,
                 title='<b>Average temperature distribution by years')
    return fig

def calculate_pop_tot(df):
    df["pop_tot"] = df["Total Population - Male"] + df["Total Population - Female"]
    return df
   
def country_emission_chart(df, year, length=30):
    df_copy = dataset_fixing(df)
    # df_copy['Area'] = df_copy['Area'].apply(get_iso_alpha)

    # plot = df_copy.loc[df_copy["Year"] == year]
    plot = df_copy.groupby('Area').mean()
    plot = plot.sort_values(by="total_emission", ascending=True).tail(length)
    colors = plt.cm.get_cmap('plasma', len(plot))
    
    plt.figure(figsize=(12, 16))
    plt.barh(plot.index, 
            #  plot['Area'],
             plot['total_emission'], 
             color=colors(range(len(plot))))
    
    plt.title(f'CO2 Emissions in top {length} countries')
    plt.xlabel('CO2 Emission in kilotones')
    
    return plt

# Define the percapita_emission function as before
def percapita_emission_chart(df, year=2020, length=30):
    df_copy = dataset_fixing(df)
    # df_copy['Area'] = df_copy['Area'].apply(get_iso_alpha)
    df_copy = calculate_pop_tot(df_copy)
     
    df_copy["per_capita_emission_kt"] = df_copy["total_emission"] / df_copy["pop_tot"]

    plot = df_copy.loc[(df_copy["Year"] == year) & (df_copy["pop_tot"] > 800000)]
    plot = plot.sort_values(by="per_capita_emission_kt", ascending=True).tail(length)
    colors = plt.cm.get_cmap('viridis', len(plot))

    plt.figure(figsize=(12, 16))
    plt.barh(plot['Area'],
             plot['per_capita_emission_kt'],
             color=colors(range(len(plot))))

    plt.title(f'CO2 agrifood per capita Emission by top {length} countries')
    return plt

# Miscellaneous helpers ---------------------------------

def get_iso_alpha(country_name):
        try:
            country = pycountry.countries.get(name=country_name)
            iso_alpha = country.alpha_3
            return iso_alpha
        except:
            return None

def dataset_fixing(df):
    temp2 = df.copy()

    temp2['Area'].replace({'United States of America': 'United States'}, inplace=True)
    temp2['Area'].replace({'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom'}, inplace=True)
    temp2['Area'].replace({'Democratic Republic of the Congo': 'Congo, The Democratic Republic of the'}, inplace=True)
    temp2['Area'].replace({'Bolivia (Plurinational State of)': 'Bolivia, Plurinational State of'}, inplace=True)
    temp2['Area'].replace({'Venezuela (Bolivarian Republic of)': 'Venezuela, Bolivarian Republic of'}, inplace=True)
    temp2['Area'].replace({'United Republic of Tanzania': 'Tanzania, United Republic of'}, inplace=True)
    temp2['Area'].replace({'Iran (Islamic Republic of)': 'Iran, Islamic Republic of'}, inplace=True)
    temp2['Area'].replace({"Democratic People's Republic of Korea": "Korea, Democratic People's Republic of"}, inplace=True)
    temp2['Area'].replace({"Republic of Korea": "Korea, Republic of"}, inplace=True)

    return temp2

# Global Visualization 1

def global_temp_inc(df,year):
    temp2 = dataset_fixing(df)

    CO2_df = temp2[['Area', 'Average Temperature °C']]
    # CO2_df = temp2.loc[df["Year"] == year]
    mean_CO2_df = CO2_df.groupby('Area').mean()

    scaler = MinMaxScaler()
    mean_CO2 = scaler.fit_transform(mean_CO2_df[['Average Temperature °C']])
    normalized_emission = pd.DataFrame(mean_CO2, columns=['Average Temperature °C'], index=mean_CO2_df.index)
    normalized_emission['Area'] = normalized_emission.index
    normalized_emission['iso_alpha'] = normalized_emission['Area'].apply(get_iso_alpha)
    normalized_emission['Average_temperature'] = normalized_emission['Average Temperature °C'].fillna(0)
    
    return normalized_emission

def global_temp_inc_map(df):
    fig = px.choropleth(
        df,
        locations="iso_alpha",
        color="Average_temperature",
        hover_name="Area",
        title="Average Temperature Increase by Country",
        color_continuous_scale=[
            (0.0, 'rgb(255, 240, 240)'),
            (0.2, 'rgb(255, 225, 225)'),
            (0.4, 'rgb(255, 150, 150)'),
            (0.6, 'rgb(255, 75, 75)'),
            (0.8, 'rgb(255, 50, 50)'),
            (1.0, 'rgb(200, 0, 0)')
        ]
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=""
        )
    )
    return fig

# Global Visualization 2

def global_emissions(df,year):
    temp_df = dataset_fixing(df)

    CO2_df = temp_df[['Area', 'total_emission']]
    # CO2_df = temp_df.loc[df["Year"] == year]
    mean_CO2_df = CO2_df.groupby('Area').mean()

    scaler = MinMaxScaler()
    mean_CO2 = scaler.fit_transform(mean_CO2_df[['total_emission']])
    normalized_emission = pd.DataFrame(mean_CO2, columns=['mean_CO2_emission'], index=mean_CO2_df.index)
    normalized_emission['Area'] = normalized_emission.index
    normalized_emission['iso_alpha'] = normalized_emission['Area'].apply(get_iso_alpha)

    return normalized_emission

def global_emissions_map(df):
    fig = px.choropleth(
        df,
        locations="iso_alpha",
        color="mean_CO2_emission",
        hover_name="Area",
        title="Average CO2 Emissions by Country",
        color_continuous_scale=[
            (0.0, 'rgb(240, 255, 240)'),
            (0.2, 'rgb(200, 255, 200)'),
            (0.4, 'rgb(150, 255, 150)'),
            (0.6, 'rgb(75, 200, 75)'),
            (0.8, 'rgb(0, 150, 0)'),
            (1.0, 'rgb(0, 100, 0)')
        ]
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=""
        )
    )
    return fig

# Global Visualization 3

def replace_outliers(df, column_name, replacement_value):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column_name] = np.where((df[column_name] < lower_bound) | (df[column_name] > upper_bound), replacement_value, df[column_name])

def per_capita_emissions(df,year=2020):
    temp_df = dataset_fixing(df)

    temp_df["total_population"] = temp_df["Total Population - Male"] + temp_df["Total Population - Female"]

    CO2_df = temp_df[['Area', 'total_emission', 'total_population']]
    mean_CO2_df = CO2_df.groupby('Area').mean()

    mean_CO2_df['per_capita_emission'] = mean_CO2_df['total_emission'] / mean_CO2_df['total_population']

    replace_outliers(mean_CO2_df, 'per_capita_emission', replacement_value = mean_CO2_df['per_capita_emission'].median())

    scaler = MinMaxScaler()
    mean_CO2 = scaler.fit_transform(mean_CO2_df[['per_capita_emission']])
    normalized_emission = pd.DataFrame(mean_CO2, columns=['mean_per_capita_emission'], index=mean_CO2_df.index)
    normalized_emission['Area'] = normalized_emission.index
    normalized_emission['iso_alpha'] = normalized_emission['Area'].apply(get_iso_alpha)

    return normalized_emission

def per_capita_emissions_map(df):
    fig = px.choropleth(
        df,
        locations="iso_alpha",
        color="mean_per_capita_emission",
        hover_name="Area",
        title="Average Per Capita Emissions by Country",
        color_continuous_scale=[
            (0.0, 'rgb(240, 240, 255)'),
            (0.2, 'rgb(150, 150, 255)'),
            (0.4, 'rgb(0, 0, 100)'),
            (0.6, 'rgb(0, 0, 70)'),
            (0.8, 'rgb(0, 0, 50)'),
            (1.0, 'rgb(0, 0, 20)')
        ]
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=""
        )
    )
    return fig

# Insights section graphs-------------------------------------------------

def continental_emission(df, year=2020):
    df["pop_tot"] = df["Total Population - Male"] + df["Total Population - Female"]
    df["continent"] = df["Area"].apply(assign_continent)
    df["per_capita_emission_kt"] = df["total_emission"] / df["pop_tot"]

    continent_df = df.loc[(df["pop_tot"] > 500000) & (df["Year"] == year)]\
        .groupby("continent")\
        .agg({"total_emission": "sum",
              "Average Temperature °C": "median",
              "per_capita_emission_kt": "mean"}).reset_index()
    continent_df = continent_df.sort_values(by="total_emission", ascending=False)
    colors = plt.cm.get_cmap('viridis', len(continent_df))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].bar(continent_df["continent"], continent_df.total_emission, color="blue")
    ax[0].set_title(f"Total CO2 emissions in {year}")

    ax2 = ax[0].twinx()                                 
    ax2.plot(continent_df["continent"], continent_df["Average Temperature °C"], color='green', marker='o')
    ax2.legend(["increasing avg temperature C°"], loc='upper right') 

    continent_df = continent_df.sort_values(by="per_capita_emission_kt", ascending=False)
    ax[1].bar(continent_df["continent"], continent_df["per_capita_emission_kt"], color="red")
    ax[1].set_title(f"Total CO2 per capita emissions in {year}")
    ax3 = ax[1].twinx() 
    ax3.plot(continent_df["continent"], continent_df["Average Temperature °C"], color='green', marker='o')
    ax3.legend(["increasing avg temperature C°"], loc='upper right')

    for axis in ax:
        axis.set_xticklabels(axis.get_xticklabels(), rotation='vertical')

    return plt  # Return the Matplotlib figure

def assign_continent(country):
    for continent, countries in continent_mapping.items():
        if country in countries:
            return continent
    return None

continent_mapping = {
    'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Côte d\'Ivoire', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe', 'Democratic Republic of the Congo', 'Ethiopia PDR', 'United Republic of Tanzania'],

    'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Russia', 'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen', 'China, Hong Kong SAR', 'China, Macao SAR', 'China, Taiwan Province of', "Democratic People's Republic of Korea", 'Iran (Islamic Republic of)', 'Republic of Korea'],

    'Europe': ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium-Luxembourg', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City', 'British Virgin Islands', 'Holy See', 'Netherlands (Kingdom of the)', 'Republic of Moldova', 'United Kingdom of Great Britain and Northern Ireland'],

    'North America': ['Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States', 'United States of America', 'United States Virgin Islands'],

    'Oceania': ['Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu', 'Micronesia (Federated States of)', 'Wallis and Futuna Islands'],
    
    'South America': ['Argentina', 'Bolivia (Plurinational State of)', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela', 'Venezuela (Bolivarian Republic of)']
}

# Clustering function
def perform_clustering(df, clus):
    mapping = {0: 'one', 1: 'two', 2: 'three', 3: 'four', 4: 'five', 5:"six", 6:"seven", 7: "eight"}

    df_2020 = dataset_fixing(df)
    df_2020 = df_2020[df_2020['Year']==2020] # focus on 2020 data
    df_2020 = df_2020.dropna(axis=1) # Drop columns with null values  
    df_2020 = df_2020.reset_index(drop=True)    
    df_clus = df_2020.drop(['Area', 'Year'], axis=1) # Drop columns 'Area' and 'Year'
  
    sc = StandardScaler()
    df_sc = sc.fit_transform(df_clus)
    df_sc = pd.DataFrame(df_sc, columns=df_clus.columns)
    
    model = KMeans(n_clusters=clus, random_state=1)
    model.fit(df_sc)
    
    cluster = model.labels_

    df_clus['Country'] = df_2020['Area']
    df_clus['Cluster'] = cluster
    
    df_clus["iso"] = df_clus["Country"].apply(get_iso_alpha)
    df_clus['Cluster'] = df_clus['Cluster'].map(mapping)
    
    return df_clus

def cluster_map(df):

    fig = px.choropleth(
        df,
        locations="iso",
        color="Cluster",
        hover_name="Country",
        title="Testing",
        # color_discrete_map = color_mapping,
    )
    return fig

clus_col=['Rice Cultivation', 'Drained organic soils (CO2)',
       'Pesticides Manufacturing', 'Food Transport', 'Food Retail',
       'On-farm Electricity Use', 'Food Packaging',
       'Agrifood Systems Waste Disposal', 'Food Processing',
       'Fertilizers Manufacturing', 'Manure left on Pasture',
       'Fires in organic soils', 'Rural population', 'Urban population',
       'Total Population - Male', 'Total Population - Female',
       'Average Temperature °C']

def cluster_scatterplot(data, cluster_col):
    fig = plt.figure(figsize=(10, 40))

    for i in range(len(clus_col)):
        plt.subplot(9, 2, i + 1)
        sns.scatterplot(data, x=data[clus_col[i]], y=data['total_emission'], hue=data[cluster_col], palette="deep", s=100)
        plt.title(clus_col[i], fontsize=14)

    plt.tight_layout()

    return fig

clustering_insights = """
# Insights based on 6 clusters:
As we see in the above map and scatterplots:
- **Clusters 1 and 6** includes countries such as **Russia**, **Europe**, most of **Africa**, **Canada** and a few others. These are the countries with the **least emissions** along with **least involvment in agricultural activities**. However, **Europe and Russia's** temperature rises sets them apart.
- **USA and Brazil** are members of **cluster 5** having moderate emissions. We get to know from the **scatter plots** that these are the nations having a higher number of **pesticide manufacturing** than the others.
- **India** is placed in **3rd cluster** with moderate emissions and a significantly higher **rural population**.
- **Indonesia** just like any other nation in cluster 1 is placed in **cluster 5**, the differentiator being it's **drained organic soils** and the **fires** taking place in them
- Finally, **China** is a country with the **highest total emissions** along with the **highest involvement in agricultural activities**.
"""
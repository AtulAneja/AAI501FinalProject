import pandas as pd
import numpy as np

# Annual trade data tables
WORLD_TRADE_DATAFILE     = '34_years_world_export_import_dataset.csv'
CPI_DATAFILE             = 'cpi_per_country.csv'
GDP_DATAFILE             = 'GDP1960-2023.csv'

# Class to help load and normalize multiple datasets into one core dataset
def load_normalized_annual_world_trade_data():
    # Read in and Merge Tables:
    base_df = pd.read_csv(WORLD_TRADE_DATAFILE)
    cpi_df = pd.read_csv(CPI_DATAFILE)
    gdp_df = pd.read_csv(GDP_DATAFILE)

    # Normalize names 
    rename_dict = {
        "Yemen, Rep."                   : "Yemen",
        "Viet Nam"                      : "Vietnam",
        "Turkiye"                       : "Turkey",
        "Hong Kong SAR, China"          : "Hong Kong, China",
        "Serbia, FR(Serbia/Montenegro)" : "Serbia", 
        "Ethiopia"                      : "Ethiopia(excludes Eritrea)",
        "Czechia"                       : "Czech Republic", 
    }


    # Filter for only the last 20ish years
    df = base_df[(base_df['Year'] >= 2000) & (base_df['Year'] <= 2023)]
    cpi_df = cpi_df[(cpi_df['Year'] >= 2000) & (cpi_df['Year'] <= 2023)]

    # Sort first by 'country' (ascending) then by 'year' (ascending)
    df = df.sort_values(by=['Partner Name', 'Year'], ascending=[True, True])

    # Normalize Column headers
    df.rename(columns={'Partner Name': 'Country'}, inplace=True)
    cpi_df.rename(columns={'Entity': 'Country'}, inplace=True)
    gdp_df.rename(columns={'Country Name': 'Country'}, inplace=True)
    gdp_df.rename(columns={'Country Code': 'Code'}, inplace=True)

    # Normalize names 
    gdp_df['Country'] = gdp_df['Country'].replace(rename_dict)
    countries = gdp_df[['Country', 'Code']].drop_duplicates()


    # Transpose table
    gdp_df = gdp_df.drop(['Code', 'Indicator Name', 'Indicator Code'], axis=1)
    df_melted = gdp_df.melt(id_vars='Country', var_name='Year', value_name='GDP')

    # Convert and merge all tables into one core table
    df_melted['Year'] = df_melted['Year'].astype(int)
    merged_df = pd.merge(df, df_melted, on=['Country', 'Year'], how='left')
    merged_df = pd.merge(merged_df, countries, on=['Country'], how='left')  

    merged_df = pd.merge(merged_df, cpi_df[['Code', 'Year', 'Consumer price index (2010 = 100)']], on=['Code', 'Year'], how='left')

    # Drop duplicate columns
    countries_only = merged_df.dropna(subset=['Code'])
    countries_only.sort_values(['Country', 'Year'], inplace=True)

    # Simplify column headers
    countries_only.rename(columns={
        'Export (US$ Thousand)' : 'Exports',
        'Import (US$ Thousand)' : 'Imports',
        'Consumer price index (2010 = 100)' : 'CPI',
        'AHS Weighted Average (%)': 'Tariff_Rate'}, inplace=True)

    #print(countries_only)

    selected_cols = ['Country','Year','Exports', 'Imports','Tariff_Rate','CPI', 'GDP']

    df = countries_only[selected_cols]
    return df
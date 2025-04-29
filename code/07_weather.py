#Data aggregation/exploration of weather during the 2 years (Didn't end up using, has potential future use)

# Import datasets with correctly formatted file paths (note this only includes trimester 1 data, if want future use make sure to include all trimesters)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df_2023 = pd.read_csv(r'data\84321 2023-08-24 to 2023-11-27 (1).csv')
df_2024 = pd.read_csv(r'data\84321 2024-08-22 to 2024-11-18 (1).csv')

print(df_2023.head())
print(df_2023.columns)

# Calculate the number of days for each DataFrame
num_days_2023 = len(df_2023)
num_days_2024 = len(df_2024)

# Find the maximum and minimum temperatures for each year
max_temp_2023 = df_2023['tempmax'].max()
min_temp_2023 = df_2023['tempmin'].min()
max_temp_2024 = df_2024['tempmax'].max()
min_temp_2024 = df_2024['tempmin'].min()
avg_temp_2023 = df_2023['temp'].mean()
avg_temp_2024 = df_2024['temp'].mean()
avg_humidity_2023 = df_2023['humidity'].mean()
avg_humidity_2024 = df_2024['humidity'].mean()
total_precipitation_2023 = df_2023['precip'].sum()
total_precipitation_2024 = df_2024['precip'].sum()
total_snow_2023 = df_2023['snow'].sum()
total_snow_2024 = df_2024['snow'].sum()
snow_days_2023 = len(df_2023[df_2023['snow'] > 0])
snow_days_2024 = len(df_2024[df_2024['snow'] > 0])
rain_days_2023 = len(df_2023[df_2023['precip'] > 0])
rain_days_2024 = len(df_2024[df_2024['precip'] > 0])

# Create a DataFrame to store the weather statistics
weather_stats = pd.DataFrame({
    'Year': [2023, 2024],
    'Number of Days': [num_days_2023, num_days_2024],
    'Max Temperature (°C)': [max_temp_2023, max_temp_2024],
    'Min Temperature (°C)': [min_temp_2023, min_temp_2024],
    'Average Temperature (°C)': [avg_temp_2023, avg_temp_2024],
    'Average Humidity (%)': [avg_humidity_2023, avg_humidity_2024],
    'Total Precipitation (mm)': [total_precipitation_2023, total_precipitation_2024],
    'Total Snow (mm)': [total_snow_2023, total_snow_2024],
    'Snow Days': [snow_days_2023, snow_days_2024],
    'Rain Days': [rain_days_2023, rain_days_2024]
})

# Display the weather statistics
print(weather_stats)

# Create visualizations

# Bar chart for average temperature
plt.figure(figsize=(8, 6))
sns.barplot(x='Year', y='Average Temperature (°C)', data=weather_stats)
plt.title('Average Temperature Comparison')
plt.show()

# Bar chart for average humidity
plt.figure(figsize=(8, 6))
sns.barplot(x='Year', y='Average Humidity (%)', data=weather_stats)
plt.title('Average Humidity Comparison')
plt.show()

# Bar chart for total precipitation
plt.figure(figsize=(8, 6))
sns.barplot(x='Year', y='Total Precipitation (mm)', data=weather_stats)
plt.title('Total Precipitation Comparison')
plt.show()

# Bar chart for total snowfall
plt.figure(figsize=(8, 6))
sns.barplot(x='Year', y='Total Snow (mm)', data=weather_stats)
plt.title('Total Snow Comparison')
plt.show()


# Bar chart for snow days
plt.figure(figsize=(8, 6))
sns.barplot(x='Year', y='Snow Days', data=weather_stats)
plt.title('Snow Days Comparison')
plt.show()

# Bar chart for rain days
plt.figure(figsize=(8, 6))
sns.barplot(x='Year', y='Rain Days', data=weather_stats)
plt.title('Rain Days Comparison')
plt.show()

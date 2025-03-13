#Work in progress

import pandas as pd
import matplotlib.pyplot as plt

# Load attendance data
gc_2023 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

# Load demographic data with dtype fix
demographics = pd.read_csv('data/modeling_data.csv', dtype=str, low_memory=False)
demographics.rename(columns={'student_number': 'Number'}, inplace=True)

# Merge attendance data with demographics
for df_name, df in zip(["gc_2023", "sv_2023", "gc_2024", "sv_2024"],
                         [gc_2023, sv_2023, gc_2024, sv_2024]):
    if 'number' in df.columns:  # Rename if lowercase
        df.rename(columns={'number': 'Number'}, inplace=True)
    df['Number'] = df['Number'].astype(str)  # Ensure consistency
    locals()[df_name] = df.merge(demographics, on='Number', how='left')

# Columns for averages (attendance columns)
columns = [1, 2, 3, 4, 5]

gc_2023_avg = gc_2023[columns].mean()
sv_2023_avg = sv_2023[columns].mean()
gc_2024_avg = gc_2024[columns].mean()
sv_2024_avg = sv_2024[columns].mean()
# 2023 T1 days: 60, 2024 T1 days: 59

# Define the three demographic columns to analyze
demographic_columns = ['gender_m', 'white_y', 'migrant_y']
years = [2023, 2024]

# Convert demographic columns to numeric in each dataset
for demo in demographic_columns:
    sv_2023[demo] = pd.to_numeric(sv_2023[demo], errors='coerce')
    gc_2023[demo] = pd.to_numeric(gc_2023[demo], errors='coerce')
    sv_2024[demo] = pd.to_numeric(sv_2024[demo], errors='coerce')
    gc_2024[demo] = pd.to_numeric(gc_2024[demo], errors='coerce')

# Compute single-value differences for each demographic (SV - GC)
diff_2023 = {}
diff_2024 = {}
for demo in demographic_columns:
    diff_2023[demo] = sv_2023[demo].mean() - gc_2023[demo].mean()
    diff_2024[demo] = sv_2024[demo].mean() - gc_2024[demo].mean()

# Create a separate figure (bar chart) for each demographic
for demo in demographic_columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Prepare the y-values for 2023 and 2024
    y_values = [diff_2023[demo], diff_2024[demo]]
    
    # Plot a bar chart for the current demographic
    ax.bar(years, y_values, color=['blue', 'orange'], width=0.4)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(f"{demo} (SV - GC)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Difference in Absences")
    
    fig.tight_layout()
    plt.show()

###################################################
# Define the three demographic columns to analyze
demographic_columns = ['gender_m', 'white_y', 'migrant_y']
years = [2023, 2024]

#this code dives into demographic exploration for T1/T2 and works to futher see the differences between them 

# Load Libraries


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------
# Load Attendance and Demographic Data
#----------------------------------------

# Load attendance data
gc_2023_t1 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t1 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t1 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t1 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

gc_2023_t2 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t2 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t2 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t2 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='SV - Absences')

# Load demographics
demographics = pd.read_csv('./data/Attendance Demographics 2024-2025.csv', dtype=str, low_memory=False)
demographics.rename(columns={'Student Number': 'Number'}, inplace=True)

#----------------------------------------
# Merge Attendance with Demographics
#----------------------------------------

datasets = {
    'gc_2023_t1': gc_2023_t1, 'sv_2023_t1': sv_2023_t1, 'gc_2024_t1': gc_2024_t1, 'sv_2024_t1': sv_2024_t1,
    'gc_2023_t2': gc_2023_t2, 'sv_2023_t2': sv_2023_t2, 'gc_2024_t2': gc_2024_t2, 'sv_2024_t2': sv_2024_t2
}

for name, df in datasets.items():
    if 'number' in df.columns:
        df.rename(columns={'number': 'Number'}, inplace=True)
    df['Number'] = df['Number'].astype(str)
    df = df.merge(demographics, on='Number', how='left')
    df['Grade Level'] = pd.to_numeric(df['Grade Level'], errors='coerce')
    datasets[name] = df

# Unpack
gc_2023_t1, sv_2023_t1, gc_2024_t1, sv_2024_t1 = datasets['gc_2023_t1'], datasets['sv_2023_t1'], datasets['gc_2024_t1'], datasets['sv_2024_t1']
gc_2023_t2, sv_2023_t2, gc_2024_t2, sv_2024_t2 = datasets['gc_2023_t2'], datasets['sv_2023_t2'], datasets['gc_2024_t2'], datasets['sv_2024_t2']

#----------------------------------------
# Naive Average Absence Graphs (T1 + T2)
#----------------------------------------

# Prepare data
data = {
    'School': ['GC', 'SV', 'GC', 'SV'],
    'Year': ['2023', '2023', '2024', '2024'],
    'Trimester': ['T1', 'T1', 'T1', 'T1'],
    'Average Absences': [gc_2023_t1['Total'].mean(), sv_2023_t1['Total'].mean(), gc_2024_t1['Total'].mean(), sv_2024_t1['Total'].mean()]
}
df_plot = pd.DataFrame(data)

data_t2 = {
    'School': ['GC', 'SV', 'GC', 'SV'],
    'Year': ['2023', '2023', '2024', '2024'],
    'Trimester': ['T2', 'T2', 'T2', 'T2'],
    'Average Absences': [gc_2023_t2['Total'].mean(), sv_2023_t2['Total'].mean(), gc_2024_t2['Total'].mean(), sv_2024_t2['Total'].mean()]
}
df_plot_t2 = pd.DataFrame(data_t2)

df_plot = pd.concat([df_plot, df_plot_t2])

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Year', y='Average Absences', hue='School', data=df_plot, palette=['blue', 'orange'])
plt.title('Naive Comparison of Average Absences Between Schools (T1 and T2)')
plt.ylabel('Average Absences')
plt.xlabel('Year')
plt.legend(title='School')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

#----------------------------------------
# Simple Filter by Demographic (Example: Race)
#----------------------------------------

filter_column = 'American Indian'  # Change as needed

# Filter
gc_2023_filtered = gc_2023_t1[gc_2023_t1[filter_column] == 'Y']
sv_2023_filtered = sv_2023_t1[sv_2023_t1[filter_column] == 'Y']
gc_2024_t1_filtered = gc_2024_t1[gc_2024_t1[filter_column] == 'Y']
sv_2024_t1_filtered = sv_2024_t1[sv_2024_t1[filter_column] == 'Y']
gc_2024_t2_filtered = gc_2024_t2[gc_2024_t2[filter_column] == 'Y']
sv_2024_t2_filtered = sv_2024_t2[sv_2024_t2[filter_column] == 'Y']
gc_2023_t2_filtered = gc_2023_t2[gc_2023_t2[filter_column] == 'Y']
sv_2023_t2_filtered = sv_2023_t2[sv_2023_t2[filter_column] == 'Y']

# Calculate means and DIDs
gc_mean_23_24_filtered = gc_2023_filtered['Total'].mean()
gc_mean_24_25_t1_filtered = gc_2024_t1_filtered['Total'].mean()
sv_mean_23_24_filtered = sv_2023_filtered['Total'].mean()
sv_mean_24_25_t1_filtered = sv_2024_t1_filtered['Total'].mean()
gc_mean_24_25_t2_filtered = gc_2024_t2_filtered['Total'].mean()
sv_mean_24_25_t2_filtered = sv_2024_t2_filtered['Total'].mean()
gc_mean_23_24_t2_filtered = gc_2023_t2_filtered['Total'].mean()
sv_mean_23_24_t2_filtered = sv_2023_t2_filtered['Total'].mean()

gc_diff_t1_filtered = gc_mean_24_25_t1_filtered - gc_mean_23_24_filtered
sv_diff_t1_filtered = sv_mean_24_25_t1_filtered - sv_mean_23_24_filtered
did_t1_filtered = gc_diff_t1_filtered - sv_diff_t1_filtered

gc_diff_t2_filtered = gc_mean_24_25_t2_filtered - gc_mean_23_24_t2_filtered
sv_diff_t2_filtered = sv_mean_24_25_t2_filtered - sv_mean_23_24_t2_filtered
did_t2_filtered = gc_diff_t2_filtered - sv_diff_t2_filtered

#----------------------------------------
# Graph Demographic DID Results
#----------------------------------------

filtered_did_df = pd.DataFrame({
    'Trimester': ['T1', 'T2'],
    'DID (Filtered)': [did_t1_filtered, did_t2_filtered]
})

plt.figure(figsize=(6, 5))
sns.barplot(x='Trimester', y='DID (Filtered)', data=filtered_did_df, palette='muted')
plt.axhline(0, color='black', linestyle='--')
plt.title(f'Difference-in-Difference by {filter_column} Students')
plt.ylabel('DID in Average Absences')
plt.xlabel('Trimester')
for i, row in filtered_did_df.iterrows():
    plt.text(i, row['DID (Filtered)'] + 0.05, f"{row['DID (Filtered)']:.2f}", ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#----------------------------------------
# Grade-Level DID Analysis (Trimester 1 and 2)
#----------------------------------------

# Collect results
t1_grade_results = []
for grade in range(9, 13):
    row = {'Grade': grade, 'Trimester': 'T1'}
    gc_23 = gc_2023_t1[gc_2023_t1['Grade Level'] == grade]
    gc_24 = gc_2024_t1[gc_2024_t1['Grade Level'] == grade]
    sv_23 = sv_2023_t1[sv_2023_t1['Grade Level'] == grade]
    sv_24 = sv_2024_t1[sv_2024_t1['Grade Level'] == grade]

    row['Total DID'] = (gc_24['Total'].mean() - gc_23['Total'].mean()) - (sv_24['Total'].mean() - sv_23['Total'].mean())

    for period in range(1, 6):
        gc_diff = gc_24[period].mean() - gc_23[period].mean()
        sv_diff = sv_24[period].mean() - sv_23[period].mean()
        row[f'Period {period} DID'] = gc_diff - sv_diff

    t1_grade_results.append(row)

t2_grade_results = []
for grade in range(9, 13):
    row = {'Grade': grade, 'Trimester': 'T2'}
    gc_23 = gc_2023_t2[gc_2023_t2['Grade Level'] == grade]
    gc_24 = gc_2024_t2[gc_2024_t2['Grade Level'] == grade]
    sv_23 = sv_2023_t2[sv_2023_t2['Grade Level'] == grade]
    sv_24 = sv_2024_t2[sv_2024_t2['Grade Level'] == grade]

    row['Total DID'] = (gc_24['Total'].mean() - gc_23['Total'].mean()) - (sv_24['Total'].mean() - sv_23['Total'].mean())

    for period in range(1, 6):
        gc_diff = gc_24[period].mean() - gc_23[period].mean()
        sv_diff = sv_24[period].mean() - sv_23[period].mean()
        row[f'Period {period} DID'] = gc_diff - sv_diff

    t2_grade_results.append(row)

# Combine all
grade_did_df = pd.DataFrame(t1_grade_results + t2_grade_results)

# Show Table
print("\n=== Grade-Level DID Summary ===")
display(grade_did_df.round(2))


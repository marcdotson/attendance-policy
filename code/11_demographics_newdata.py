import pandas as pd

# Load attendance data for both trimesters
gc_2023 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t1 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t1 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

gc_2024_t2 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='GC - Absences')  # Second trimester
sv_2024_t2 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='SV - Absences')  # Second trimester

# Load the second trimester data for 2023
gc_2023_t2 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='GC - Absence')  # Second trimester 2023
sv_2023_t2 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='SV - Absence')  # Second trimester 2023

# Load the new demographics data
demographics = pd.read_csv('./data/Attendance Demographics 2024-2025.csv', dtype=str, low_memory=False)

# Rename 'Student Number' to 'Number' to match other dataframes
demographics.rename(columns={'Student Number': 'Number'}, inplace=True)

# Merge attendance data with demographics for all periods
for df_name, df in zip(
    ["gc_2023", "sv_2023", "gc_2024_t1", "sv_2024_t1", "gc_2024_t2", "sv_2024_t2", "gc_2023_t2", "sv_2023_t2"],
    [gc_2023, sv_2023, gc_2024_t1, sv_2024_t1, gc_2024_t2, sv_2024_t2, gc_2023_t2, sv_2023_t2]
):
    if 'number' in df.columns:  # Rename if lowercase
        df.rename(columns={'number': 'Number'}, inplace=True)
    df['Number'] = df['Number'].astype(str)  # Ensure consistency
    locals()[df_name] = df.merge(demographics, on='Number', how='left')

#######################################################################
# Input column you want to filter 
filter_column = 'American Indian'
#######################################################################

# Filter the datasets where the specified column has a value of 'Y'
gc_2023_filtered = gc_2023[gc_2023[filter_column] == 'Y']
sv_2023_filtered = sv_2023[sv_2023[filter_column] == 'Y']
gc_2024_t1_filtered = gc_2024_t1[gc_2024_t1[filter_column] == 'Y']
sv_2024_t1_filtered = sv_2024_t1[sv_2024_t1[filter_column] == 'Y']
gc_2024_t2_filtered = gc_2024_t2[gc_2024_t2[filter_column] == 'Y']
sv_2024_t2_filtered = sv_2024_t2[sv_2024_t2[filter_column] == 'Y']
gc_2023_t2_filtered = gc_2023_t2[gc_2023_t2[filter_column] == 'Y']
sv_2023_t2_filtered = sv_2023_t2[sv_2023_t2[filter_column] == 'Y']

# Calculate the means for the filtered datasets for both trimesters
gc_mean_23_24_filtered = gc_2023_filtered['Total'].mean()
gc_mean_24_25_t1_filtered = gc_2024_t1_filtered['Total'].mean()
sv_mean_23_24_filtered = sv_2023_filtered['Total'].mean()
sv_mean_24_25_t1_filtered = sv_2024_t1_filtered['Total'].mean()
gc_mean_24_25_t2_filtered = gc_2024_t2_filtered['Total'].mean()
sv_mean_24_25_t2_filtered = sv_2024_t2_filtered['Total'].mean()
gc_mean_23_24_t2_filtered = gc_2023_t2_filtered['Total'].mean()
sv_mean_23_24_t2_filtered = sv_2023_t2_filtered['Total'].mean()

print(f'Filtered by {filter_column} results:')

# Calculate the DID for the filtered data for both trimesters
gc_diff_t1_filtered = gc_mean_24_25_t1_filtered - gc_mean_23_24_filtered
sv_diff_t1_filtered = sv_mean_24_25_t1_filtered - sv_mean_23_24_filtered
did_t1_filtered = gc_diff_t1_filtered - sv_diff_t1_filtered

gc_diff_t2_filtered = gc_mean_24_25_t2_filtered - gc_mean_23_24_t2_filtered
sv_diff_t2_filtered = sv_mean_24_25_t2_filtered - sv_mean_23_24_t2_filtered
did_t2_filtered = gc_diff_t2_filtered - sv_diff_t2_filtered

# Print the DID for the filtered data for each trimester
print(f'DID in mean total absences for 1st trimester is {did_t1_filtered:.2f}')
print(f'DID in mean total absences for 2nd trimester is {did_t2_filtered:.2f}')

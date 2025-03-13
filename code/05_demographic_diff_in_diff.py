import pandas as pd

#load attendance data
gc_2023 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

#load demographic data (if necessary)
demographics = pd.read_csv('./data/modeling_data.csv', dtype=str, low_memory=False)
demographics.rename(columns={'student_number': 'Number'}, inplace=True)

#merge attendance data with demographics
for df_name, df in zip(["gc_2023", "sv_2023", "gc_2024", "sv_2024"],
                         [gc_2023, sv_2023, gc_2024, sv_2024]):
    if 'number' in df.columns:  # Rename if lowercase
        df.rename(columns={'number': 'Number'}, inplace=True)
    df['Number'] = df['Number'].astype(str)  # Ensure consistency
    locals()[df_name] = df.merge(demographics, on='Number', how='left')

#define the part-time columns
part_time_cols = ['part_time_home_school_h', 'part_time_home_school_s', 'part_time_home_school_p']

#function to assign student type (creating two columns)
def categorize_student(row):
    # Convert to numeric, errors='coerce' will turn any non-numeric values into NaN
    part_time_values = pd.to_numeric(row[part_time_cols], errors='coerce')
    
    # If any part-time column has a non-zero value, classify as "Part-Time"
    if part_time_values.sum() > 0:
        return pd.Series([0, 1])  # Full-Time = 0, Part-Time = 1
    return pd.Series([1, 0])  # Full-Time = 1, Part-Time = 0


#######################################################################
#input column you want to filter 
filter_column = 'Full-Time'
#######################################################################



#apply the function to all DataFrames, creating Part-Time and Full-Time columns
for df in [gc_2023, sv_2023, gc_2024, sv_2024]:
    df[['Full-Time', 'Part-Time']] = df.apply(categorize_student, axis=1)
    df[filter_column] = pd.to_numeric(df[filter_column], errors='coerce')


#column to filter by (e.g., 'Part-Time')


#filter the datasets where the specified column has a value of 1
gc_2023_filtered = gc_2023[gc_2023[filter_column] == 1]
sv_2023_filtered = sv_2023[sv_2023[filter_column] == 1]
gc_2024_filtered = gc_2024[gc_2024[filter_column] == 1]
sv_2024_filtered = sv_2024[sv_2024[filter_column] == 1]

#calculate the means for the filtered datasets
gc_mean_23_24_filtered = gc_2023_filtered['Total'].mean()
gc_mean_24_25_filtered = gc_2024_filtered['Total'].mean()
sv_mean_23_24_filtered = sv_2023_filtered['Total'].mean()
sv_mean_24_25_filtered = sv_2024_filtered['Total'].mean()

print(f'Filerted by {filter_column} results:')

#calculate the DID for the filtered data
gc_diff_filtered = gc_mean_24_25_filtered - gc_mean_23_24_filtered
sv_diff_filtered = sv_mean_24_25_filtered - sv_mean_23_24_filtered
did_filtered = gc_diff_filtered - sv_diff_filtered

#print the DID for the filtered data
print(f'DID in mean total absences is {did_filtered:.2f}')

# Loop through the periods (1 to 5)

for period in range(1, 6):
    # Calculate mean for each period
    gc_mean_before_filtered = gc_2023_filtered[period].mean()
    gc_mean_after_filtered = gc_2024_filtered[period].mean()
    sv_mean_before_filtered = sv_2023_filtered[period].mean()
    sv_mean_after_filtered = sv_2024_filtered[period].mean()
    
    # Calculate the DID for each period
    gc_diff_filtered = gc_mean_after_filtered - gc_mean_before_filtered
    sv_diff_filtered = sv_mean_after_filtered - sv_mean_before_filtered
    did_filtered = gc_diff_filtered - sv_diff_filtered

    print(f'DID in mean total absences (period {period}) is {did_filtered:.2f}')



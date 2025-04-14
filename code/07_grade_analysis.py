import pandas as pd

#load attendance data
gc_2023 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

#load demographic data (if necessary)
demographics = pd.read_csv('./data/exploratory_data.csv', dtype=str, low_memory=False)
demographics.rename(columns={'student_number': 'Number'}, inplace=True)

#merge attendance data with demographics
for df_name, df in zip(["gc_2023", "sv_2023", "gc_2024", "sv_2024"],
                         [gc_2023, sv_2023, gc_2024, sv_2024]):
    if 'number' in df.columns:  #rename if lowercase
        df.rename(columns={'number': 'Number'}, inplace=True)
    df['Number'] = df['Number'].astype(str)  #ensure consistency
    locals()[df_name] = df.merge(demographics, on='Number', how='left')



for df_name in ["gc_2023", "sv_2023"]: #2023 years
    df = locals()[df_name]  # Retrieve the DataFrame
    
    #convert year and current_grade to numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['current_grade'] = pd.to_numeric(df['current_grade'], errors='coerce')

    #filter for year 2023
    df = df[df['year'] == 2023]


for df_name in ["gc_2024", "sv_2024"]: #2024 years
    df = locals()[df_name]  
    
    #convert year and current_grade to numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['current_grade'] = pd.to_numeric(df['current_grade'], errors='coerce')

    #filter for year 2024
    df = df[df['year'] == 2024]

    #increment current_grade (only for valid numbers)
    df.loc[df['current_grade'].notna(), 'current_grade'] += 1
    #update the variable
    locals()[df_name] = df


#loop through grade levels (9-12)
for grade in range(9, 13):
    gc_23_avg = gc_2023[gc_2023['current_grade'] == grade]['Total'].mean()
    gc_24_avg = gc_2024[gc_2024['current_grade'] == grade]['Total'].mean()
    sv_23_avg = sv_2023[sv_2023['current_grade'] == grade]['Total'].mean()
    sv_24_avg = sv_2024[sv_2024['current_grade'] == grade]['Total'].mean()

    #Difference-in-Difference (DID)
    gc_diff = gc_24_avg - gc_23_avg
    sv_diff = sv_24_avg - sv_23_avg
    did = gc_diff - sv_diff

    print(f'Grade {grade}: DID in mean total absences is {did:.2f}')

    # Loop through periods (1-5)
    for period in range(1, 6):
        gc_23_avg = gc_2023[gc_2023['current_grade'] == grade][period].mean()
        gc_24_avg = gc_2024[gc_2024['current_grade'] == grade][period].mean()
        sv_23_avg = sv_2023[sv_2023['current_grade'] == grade][period].mean()
        sv_24_avg = sv_2024[sv_2024['current_grade'] == grade][period].mean()

        gc_diff = gc_24_avg - gc_23_avg
        sv_diff = sv_24_avg - sv_23_avg
        did = gc_diff - sv_diff

        print(f'Grade {grade}: DID in mean absences (period {period}) is {did:.2f}')
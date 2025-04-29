#This document looks into T1 demographic data and how it related to absenses per period 

import pandas as pd

def load_data():
    # Load attendance data
    gc_2023 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name= 'GC - Tardies')
    sv_2023 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name= 'SV - Tardies')
    gc_2024 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name= 'GC - Tardies')
    sv_2024 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name= 'SV - Tardies')
    # Load demographics
    demographics = pd.read_csv('data/Attendance Demographics 2024-2025.csv', dtype=str, low_memory=False)
    demographics.rename(columns={'Student Number': 'Number'}, inplace=True)

    return gc_2023, sv_2023, gc_2024, sv_2024, demographics
#This will work to merge the demographic file to the absense files
def merge_attendance_with_demographics(attendance_dfs, demographics):
    merged_dfs = []
    for df in attendance_dfs:
        if 'number' in df.columns:
            df.rename(columns={'number': 'Number'}, inplace=True)
        df['Number'] = df['Number'].astype(str)
        merged = df.merge(demographics, on='Number', how='left')
        merged_dfs.append(merged)
    return merged_dfs
#Category work to help prepare for diff and diff
def categorize_student(row, part_time_cols):
    part_time_values = pd.to_numeric(row[part_time_cols], errors='coerce')
    if part_time_values.sum() > 0:
        return pd.Series([0, 1])  # Full-Time = 0, Part-Time = 1
    return pd.Series([1, 0])      # Full-Time = 1, Part-Time = 0

def apply_student_categories(attendance_dfs, part_time_cols):
    for df in attendance_dfs:
        df[['Full-Time', 'Part-Time']] = df.apply(categorize_student, axis=1, args=(part_time_cols,))
        df['Full-Time'] = pd.to_numeric(df['Full-Time'], errors='coerce')
    return attendance_dfs
#Diff in diff calculation
def calculate_did(gc_2023, sv_2023, gc_2024, sv_2024):
    gc_mean_before = gc_2023['Total'].mean()
    gc_mean_after = gc_2024['Total'].mean()
    sv_mean_before = sv_2023['Total'].mean()
    sv_mean_after = sv_2024['Total'].mean()
    gc_diff = gc_mean_after - gc_mean_before
    sv_diff = sv_mean_after - sv_mean_before
    did = gc_diff - sv_diff
    print(f'DID in mean total absences: {did:.2f}')

    # Period-specific DID
    for period in range(1, 6):
        gc_before = gc_2023[period].mean()
        gc_after = gc_2024[period].mean()
        sv_before = sv_2023[period].mean()
        sv_after = sv_2024[period].mean()

        gc_period_diff = gc_after - gc_before
        sv_period_diff = sv_after - sv_before
        did_period = gc_period_diff - sv_period_diff

        print(f'DID in mean absences (period {period}): {did_period:.2f}')

def main():
    # Constants
    part_time_cols = []  # No part-time columns exist (old data file had them)

    # Load and prepare data
    gc_2023, sv_2023, gc_2024, sv_2024, demographics = load_data()
    attendance_dfs = [gc_2023, sv_2023, gc_2024, sv_2024]
    attendance_dfs = merge_attendance_with_demographics(attendance_dfs, demographics)

    gc_2023, sv_2023, gc_2024, sv_2024 = attendance_dfs  # no filtering needed

    print(f'\nAll students:')
    calculate_did(gc_2023, sv_2023, gc_2024, sv_2024)

if __name__ == "__main__":
    main()

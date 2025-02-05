import pandas as pd

gc_2023 = pd.read_excel('../data/23-24 T1 Attendance.xlsx', sheet_name= 'GC - Absence')
sv_2023 = pd.read_excel('../data/23-24 T1 Attendance.xlsx', sheet_name= 'SV - Absence')
gc_2024 = pd.read_excel('../data/24-25 T1 Attendance.xlsx', sheet_name= 'GC - Absences')
sv_2024 = pd.read_excel('../data/24-25 T1 Attendance.xlsx', sheet_name= 'SV - Absences')

# check by calculating the mean for each group directly

gc_mean_23_24 = gc_2023['Total'].mean()
gc_mean_24_25 = gc_2024['Total'].mean()
sv_mean_23_24 = sv_2023['Total'].mean()
sv_mean_24_25 = sv_2024['Total'].mean()

print(f'mean gc absences before: {gc_mean_23_24:.2f}')
print(f'mean gc absences after: {gc_mean_24_25:.2f}')
print(f'mean sv absences before: {sv_mean_23_24:.2f}')
print(f'mean sv absences after: {sv_mean_24_25:.2f}')

gc_diff = gc_mean_24_25 - gc_mean_23_24
sv_diff = sv_mean_24_25 - sv_mean_23_24
did = gc_diff - sv_diff

print(f'DID in mean total absences (not split by period) is {did:.2f}')

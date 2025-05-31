#----------------------------------------
# Diff-in-Diff of Absences: GC vs SV (T1, T2, & T3)
#----------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

#----------------------------------------
# 1. Load Data
#----------------------------------------

gc_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

gc_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='SV - Absences')

gc_2023_t3 = pd.read_excel('data/23-24 T3 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t3 = pd.read_excel('data/23-24 T3 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t3 = pd.read_excel('data/24-25 T3 Attendance.xlsx', sheet_name='GC - Absence')
sv_2024_t3 = pd.read_excel('data/24-25 T3 Attendance.xlsx', sheet_name='SV - Absence')

#----------------------------------------
# 2. Find Averages by Period
#----------------------------------------

periods = [1, 2, 3, 4, 5]

gc_2023_avg_t1 = gc_2023_t1[periods].mean()
sv_2023_avg_t1 = sv_2023_t1[periods].mean()
gc_2024_avg_t1 = gc_2024_t1[periods].mean()
sv_2024_avg_t1 = sv_2024_t1[periods].mean()

gc_2023_avg_t2 = gc_2023_t2[periods].mean()
sv_2023_avg_t2 = sv_2023_t2[periods].mean()
gc_2024_avg_t2 = gc_2024_t2[periods].mean()
sv_2024_avg_t2 = sv_2024_t2[periods].mean()

gc_2023_avg_t3 = gc_2023_t3[periods].mean()
sv_2023_avg_t3 = sv_2023_t3[periods].mean()
gc_2024_avg_t3 = gc_2024_t3[periods].mean()
sv_2024_avg_t3 = sv_2024_t3[periods].mean()

#----------------------------------------
# 3. Prepare Data for Diff-in-Diff
#----------------------------------------

# Trimester 1
# GC is the control (0) and SV is the treatment (1)
gc_2023_long_t1 = pd.DataFrame(gc_2023_avg_t1).reset_index()
gc_2023_long_t1.columns = ['period', 'avg_absences']
gc_2023_long_t1['school'] = 0
gc_2023_long_t1['year'] = 0

sv_2023_long_t1 = pd.DataFrame(sv_2023_avg_t1).reset_index()
sv_2023_long_t1.columns = ['period', 'avg_absences']
sv_2023_long_t1['school'] = 1
sv_2023_long_t1['year'] = 0

gc_2024_long_t1 = pd.DataFrame(gc_2024_avg_t1).reset_index()
gc_2024_long_t1.columns = ['period', 'avg_absences']
gc_2024_long_t1['school'] = 0
gc_2024_long_t1['year'] = 1

sv_2024_long_t1 = pd.DataFrame(sv_2024_avg_t1).reset_index()
sv_2024_long_t1.columns = ['period', 'avg_absences']
sv_2024_long_t1['school'] = 1
sv_2024_long_t1['year'] = 1

df_reg_t1 = pd.concat([gc_2023_long_t1, sv_2023_long_t1, gc_2024_long_t1, sv_2024_long_t1])
df_reg_t1['school_year'] = df_reg_t1['school'] * df_reg_t1['year']

# Trimester 2
# GC is the control (0) and SV is the treatment (1)
gc_2023_long_t2 = pd.DataFrame(gc_2023_avg_t2).reset_index()
gc_2023_long_t2.columns = ['period', 'avg_absences']
gc_2023_long_t2['school'] = 0
gc_2023_long_t2['year'] = 0

sv_2023_long_t2 = pd.DataFrame(sv_2023_avg_t2).reset_index()
sv_2023_long_t2.columns = ['period', 'avg_absences']
sv_2023_long_t2['school'] = 1
sv_2023_long_t2['year'] = 0

gc_2024_long_t2 = pd.DataFrame(gc_2024_avg_t2).reset_index()
gc_2024_long_t2.columns = ['period', 'avg_absences']
gc_2024_long_t2['school'] = 0
gc_2024_long_t2['year'] = 1

sv_2024_long_t2 = pd.DataFrame(sv_2024_avg_t2).reset_index()
sv_2024_long_t2.columns = ['period', 'avg_absences']
sv_2024_long_t2['school'] = 1
sv_2024_long_t2['year'] = 1

df_reg_t2 = pd.concat([gc_2023_long_t2, sv_2023_long_t2, gc_2024_long_t2, sv_2024_long_t2])
df_reg_t2['school_year'] = df_reg_t2['school'] * df_reg_t2['year']

# Trimester 3
# GC is the control (0) and SV is the treatment (1)
gc_2023_long_t3 = pd.DataFrame(gc_2023_avg_t3).reset_index()
gc_2023_long_t3.columns = ['period', 'avg_absences']
gc_2023_long_t3['school'] = 0
gc_2023_long_t3['year'] = 0

sv_2023_long_t3 = pd.DataFrame(sv_2023_avg_t3).reset_index()
sv_2023_long_t3.columns = ['period', 'avg_absences']
sv_2023_long_t3['school'] = 1
sv_2023_long_t3['year'] = 0

gc_2024_long_t3 = pd.DataFrame(gc_2024_avg_t3).reset_index()
gc_2024_long_t3.columns = ['period', 'avg_absences']
gc_2024_long_t3['school'] = 0
gc_2024_long_t3['year'] = 1

sv_2024_long_t3 = pd.DataFrame(sv_2024_avg_t3).reset_index()
sv_2024_long_t3.columns = ['period', 'avg_absences']
sv_2024_long_t3['school'] = 1
sv_2024_long_t3['year'] = 1

df_reg_t3 = pd.concat([gc_2023_long_t3, sv_2023_long_t3, gc_2024_long_t3, sv_2024_long_t3])
df_reg_t3['school_year'] = df_reg_t3['school'] * df_reg_t3['year']

#----------------------------------------
# 4. Calculate Diff-in-Diff
#----------------------------------------

# Trimester 1
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg_t1).fit()
print(ols_model.summary())

mean_gc_2023_t1 = gc_2023_long_t1['avg_absences'].mean()
mean_sv_2023_t1 = sv_2023_long_t1['avg_absences'].mean()
mean_gc_2024_t1 = gc_2024_long_t1['avg_absences'].mean()
mean_sv_2024_t1 = sv_2024_long_t1['avg_absences'].mean()

gc_diff_t1 = mean_gc_2024_t1 - mean_gc_2023_t1
sv_diff_t1 = mean_sv_2024_t1 - mean_sv_2023_t1
did_t1 = sv_diff_t1 - gc_diff_t1

print(f'\nManual DID Calculation:')
print(f'Mean GC Absences Before: {mean_gc_2023_t1:.2f}')
print(f'Mean SV Absences Before: {mean_sv_2023_t1:.2f}')
print(f'Mean GC Absences After: {mean_gc_2024_t1:.2f}')
print(f'Mean SV Absences After: {mean_sv_2024_t1:.2f}')
print(f'DID in Mean Absences: {did_t1:.2f}')

# Trimester 2
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg_t2).fit()
print(ols_model.summary())

mean_gc_2023_t2 = gc_2023_long_t2['avg_absences'].mean()
mean_sv_2023_t2 = sv_2023_long_t2['avg_absences'].mean()
mean_gc_2024_t2 = gc_2024_long_t2['avg_absences'].mean()
mean_sv_2024_t2 = sv_2024_long_t2['avg_absences'].mean()

gc_diff_t2 = mean_gc_2024_t2 - mean_gc_2023_t2
sv_diff_t2 = mean_sv_2024_t2 - mean_sv_2023_t2
did_t2 = sv_diff_t2 - gc_diff_t2

print(f'\nManual DID Calculation:')
print(f'Mean GC Absences Before: {mean_gc_2023_t2:.2f}')
print(f'Mean SV Absences Before: {mean_sv_2023_t2:.2f}')
print(f'Mean GC Absences After: {mean_gc_2024_t2:.2f}')
print(f'Mean SV Absences After: {mean_sv_2024_t2:.2f}')
print(f'DID in Mean Absences: {did_t2:.2f}')

# Trimester 3
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg_t3).fit()
print(ols_model.summary())

mean_gc_2023_t3 = gc_2023_long_t3['avg_absences'].mean()
mean_sv_2023_t3 = sv_2023_long_t3['avg_absences'].mean()
mean_gc_2024_t3 = gc_2024_long_t3['avg_absences'].mean()
mean_sv_2024_t3 = sv_2024_long_t3['avg_absences'].mean()

gc_diff_t3 = mean_gc_2024_t3 - mean_gc_2023_t3
sv_diff_t3 = mean_sv_2024_t3 - mean_sv_2023_t3
did_t3 = sv_diff_t3 - gc_diff_t3

print(f'\nManual DID Calculation:')
print(f'Mean GC Absences Before: {mean_gc_2023_t3:.2f}')
print(f'Mean SV Absences Before: {mean_sv_2023_t3:.2f}')
print(f'Mean GC Absences After: {mean_gc_2024_t3:.2f}')
print(f'Mean SV Absences After: {mean_sv_2024_t3:.2f}')
print(f'DID in Mean Absences: {did_t3:.2f}')


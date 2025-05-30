#----------------------------------------
# Attendance Comparison: GC vs SV (T1, T2, & T3)
#----------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

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
# 2. Find Averages by Trimester
#----------------------------------------

gc_2023_tot_t1 = gc_2023_t1['Total'].mean()
sv_2023_tot_t1 = sv_2023_t1['Total'].mean()
gc_2024_tot_t1 = gc_2024_t1['Total'].mean()
sv_2024_tot_t1 = sv_2024_t1['Total'].mean()

gc_2023_tot_t2 = gc_2023_t2['Total'].mean()
sv_2023_tot_t2 = sv_2023_t2['Total'].mean()
gc_2024_tot_t2 = gc_2024_t2['Total'].mean()
sv_2024_tot_t2 = sv_2024_t2['Total'].mean()

gc_2023_tot_t3 = gc_2023_t3['Total'].mean()
sv_2023_tot_t3 = sv_2023_t3['Total'].mean()
gc_2024_tot_t3 = gc_2024_t3['Total'].mean()
sv_2024_tot_t3 = sv_2024_t3['Total'].mean()

# Visualize absences trend
time_periods = ['2023 T1', '2023 T2', '2023 T3', '2024 T1', '2024 T2', '2024 T3']
gc_absences = [gc_2023_tot_t1, gc_2023_tot_t2, gc_2023_tot_t3, gc_2024_tot_t1, gc_2024_tot_t2, gc_2024_tot_t3]
sv_absences = [sv_2023_tot_t1, sv_2023_tot_t2, sv_2023_tot_t3, sv_2024_tot_t1, sv_2024_tot_t2, sv_2024_tot_t3]

plt.figure(figsize=(10,6))
plt.plot(time_periods, gc_absences, marker='o', linestyle='-', color='green', label='GC')
plt.plot(time_periods, sv_absences, marker='o', linestyle='-', color='blue', label='SV')

# Labels and title
plt.xlabel('Time Period')
plt.ylabel('Average Absences')
plt.title('Trend of Average Absences by School')
plt.grid(True)

# Legend
plt.legend(title='School')

#----------------------------------------
# 3. Find Averages by Period
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





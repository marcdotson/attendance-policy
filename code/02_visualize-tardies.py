#----------------------------------------
# Tardies Comparison: GC vs SV (T1, T2, & T3)
#----------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------
# 1. Load Data
#----------------------------------------

gc_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='SV - Tardies')
gc_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='SV - Tardies')

gc_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='SV - Tardies')
gc_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='SV - Tardies')

gc_2023_t3 = pd.read_excel('data/23-24 T3 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2023_t3 = pd.read_excel('data/23-24 T3 Attendance.xlsx', sheet_name='SV - Tardies')
gc_2024_t3 = pd.read_excel('data/24-25 T3 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2024_t3 = pd.read_excel('data/24-25 T3 Attendance.xlsx', sheet_name='SV - Tardies')

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

# Visualize overall tardies trend
time_periods = ['2023 T1', '2023 T2', '2023 T3', '2024 T1', '2024 T2', '2024 T3']
gc_tardies = [gc_2023_tot_t1, gc_2023_tot_t2, gc_2023_tot_t3, gc_2024_tot_t1, gc_2024_tot_t2, gc_2024_tot_t3]
sv_tardies = [sv_2023_tot_t1, sv_2023_tot_t2, sv_2023_tot_t3, sv_2024_tot_t1, sv_2024_tot_t2, sv_2024_tot_t3]

# Plot
plt.figure(figsize=(10,6))
plt.plot(time_periods, gc_tardies, marker='o', linestyle='-', color='green', label='GC')
plt.plot(time_periods, sv_tardies, marker='o', linestyle='-', color='blue', label='SV')

# Labels, title, and legend
plt.xlabel('Year and Trimester')
plt.ylabel('Average Tardies')
plt.title('Trend of Average Tardies by School')
plt.grid(True)
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

# Visualize tardies trend by period
gc_2023_avg = pd.DataFrame({
    'avg_t1': gc_2023_avg_t1.values,
    'avg_t2': gc_2023_avg_t2.values,
    'avg_t3': gc_2023_avg_t3.values
}).T.mean()

gc_2024_avg = pd.DataFrame({
    'avg_t1': gc_2024_avg_t1.values,
    'avg_t2': gc_2024_avg_t2.values,
    'avg_t3': gc_2024_avg_t3.values
}).T.mean()

sv_2023_avg = pd.DataFrame({
    'avg_t1': sv_2023_avg_t1.values,
    'avg_t2': sv_2023_avg_t2.values,
    'avg_t3': sv_2023_avg_t3.values
}).T.mean()

sv_2024_avg = pd.DataFrame({
    'avg_t1': sv_2024_avg_t1.values,
    'avg_t2': sv_2024_avg_t2.values,
    'avg_t3': sv_2024_avg_t3.values
}).T.mean()

# Plot
plt.figure(figsize=(10,6))
plt.plot(periods, gc_2023_avg.values, label='GC 23-24', marker='o', color='darkgreen')
plt.plot(periods, sv_2023_avg.values, label='SV 23-24', marker='o', color='darkblue')
plt.plot(periods, gc_2024_avg.values, label='GC 24-25', marker='o', color='lightgreen')
plt.plot(periods, sv_2024_avg.values, label='SV 24-25', marker='o', color='lightblue')

# Labels, title, and legend
plt.xlabel('Period')
plt.ylabel('Average Tardies')
plt.title('Trend of Average Tardies Per Period')
plt.grid(True)
plt.legend()


#----------------------------------------
# Tardies Comparison: GC vs SV (T1 & T2)
#----------------------------------------

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

#----------------------------------------
# 1. Load Data (Tardies)
#----------------------------------------

gc_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='SV - Tardies')
gc_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='SV - Tardies')
gc_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='SV - Tardies')
gc_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='GC - Tardies')
sv_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='SV - Tardies')

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

#----------------------------------------
# 3. Print Averages (Optional, can comment out later)
#----------------------------------------

print("GC 2023 T1 Tardy Averages:\n", gc_2023_avg_t1)
print("SV 2023 T1 Tardy Averages:\n", sv_2023_avg_t1)
print("GC 2024 T1 Tardy Averages:\n", gc_2024_avg_t1)
print("SV 2024 T1 Tardy Averages:\n", sv_2024_avg_t1)
print("GC 2023 T2 Tardy Averages:\n", gc_2023_avg_t2)
print("SV 2023 T2 Tardy Averages:\n", sv_2023_avg_t2)
print("GC 2024 T2 Tardy Averages:\n", gc_2024_avg_t2)
print("SV 2024 T2 Tardy Averages:\n", sv_2024_avg_t2)

#----------------------------------------
# 4. Calculate Differences (SV - GC)
#----------------------------------------

diff_2023_t1 = sv_2023_avg_t1 - gc_2023_avg_t1
diff_2024_t1 = sv_2024_avg_t1 - gc_2024_avg_t1
diff_2023_t2 = sv_2023_avg_t2 - gc_2023_avg_t2
diff_2024_t2 = sv_2024_avg_t2 - gc_2024_avg_t2

#----------------------------------------
# 5. Plot Differences - T1 and T2
#----------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# T1 Differences
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].plot(diff_2023_t1.index, diff_2023_t1.values, label='23-24 T1', marker='o', color='blue', linestyle='-')
axes[0].plot(diff_2024_t1.index, diff_2024_t1.values, label='24-25 T1', marker='s', color='red', linestyle='--')
axes[0].set_title('T1: SV - GC Tardy Differences', fontsize=16, pad=15)
axes[0].set_xlabel('Period', fontsize=14)
axes[0].set_ylabel('Average Tardy Difference', fontsize=14)
axes[0].legend(loc='best')
axes[0].grid(True)

# T2 Differences
axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[1].plot(diff_2023_t2.index, diff_2023_t2.values, label='23-24 T2', marker='o', color='green', linestyle='-')
axes[1].plot(diff_2024_t2.index, diff_2024_t2.values, label='24-25 T2', marker='s', color='orange', linestyle='--')
axes[1].set_title('T2: SV - GC Tardy Differences', fontsize=16, pad=15)
axes[1].set_xlabel('Period', fontsize=14)
axes[1].legend(loc='best')
axes[1].grid(True)

plt.tight_layout()
plt.show()

#----------------------------------------
# 6. Plot Average Tardies by School (T1 and T2)
#----------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

# T1 Tardies
axes[0].plot(gc_2023_avg_t1.index, gc_2023_avg_t1.values, label='GC 23-24 T1', marker='o', color='blue')
axes[0].plot(sv_2023_avg_t1.index, sv_2023_avg_t1.values, label='SV 23-24 T1', marker='o', color='green')
axes[0].plot(gc_2024_avg_t1.index, gc_2024_avg_t1.values, label='GC 24-25 T1', marker='s', color='red')
axes[0].plot(sv_2024_avg_t1.index, sv_2024_avg_t1.values, label='SV 24-25 T1', marker='s', color='orange')
axes[0].set_title('T1: Average Tardies', fontsize=16, pad=15)
axes[0].set_xlabel('Period', fontsize=14)
axes[0].set_ylabel('Average Tardies', fontsize=14)
axes[0].legend(loc='best')
axes[0].grid(True)

# T2 Tardies
axes[1].plot(gc_2023_avg_t2.index, gc_2023_avg_t2.values, label='GC 23-24 T2', marker='o', color='purple')
axes[1].plot(sv_2023_avg_t2.index, sv_2023_avg_t2.values, label='SV 23-24 T2', marker='o', color='brown')
axes[1].plot(gc_2024_avg_t2.index, gc_2024_avg_t2.values, label='GC 24-25 T2', marker='s', color='pink')
axes[1].plot(sv_2024_avg_t2.index, sv_2024_avg_t2.values, label='SV 24-25 T2', marker='s', color='black')
axes[1].set_title('T2: Average Tardies', fontsize=16, pad=15)
axes[1].set_xlabel('Period', fontsize=14)
axes[1].legend(loc='best')
axes[1].grid(True)

plt.tight_layout()
plt.show()

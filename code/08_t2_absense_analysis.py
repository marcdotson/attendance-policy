#----------------------------------------
# T2 Absence Analysis: GC vs SV
#----------------------------------------

# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

#----------------------------------------
# 1. Load T2 Data
#----------------------------------------

gc_2023 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='SV - Absences')

#----------------------------------------
# 2. Calculate Average Absences per Period
#----------------------------------------

periods = [1, 2, 3, 4, 5]

gc_2023_avg = gc_2023[periods].mean()
sv_2023_avg = sv_2023[periods].mean()
gc_2024_avg = gc_2024[periods].mean()
sv_2024_avg = sv_2024[periods].mean()

#----------------------------------------
# 3. Print Averages (Optional - comment if not needed)
#----------------------------------------

print("GC 2023 T2 Averages:\n", gc_2023_avg)
print("SV 2023 T2 Averages:\n", sv_2023_avg)
print("GC 2024 T2 Averages:\n", gc_2024_avg)
print("SV 2024 T2 Averages:\n", sv_2024_avg)

#----------------------------------------
# 4. Plot Raw Average Absences
#----------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(gc_2023_avg.index, gc_2023_avg.values, label='GC 23-24', marker='o', color='blue')
ax.plot(sv_2023_avg.index, sv_2023_avg.values, label='SV 23-24', marker='o', color='green')
ax.plot(gc_2024_avg.index, gc_2024_avg.values, label='GC 24-25', marker='s', color='red')
ax.plot(sv_2024_avg.index, sv_2024_avg.values, label='SV 24-25', marker='s', color='orange')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Average Absences', fontsize=14)
ax.set_title('Average Absences per Period (T2)', fontsize=16, pad=15)
ax.legend(loc='best')
ax.grid(True)

plt.tight_layout()
plt.show()

#----------------------------------------
# 5. Plot Proportion of Absences (Divided by Days)
#----------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

# Normalize absences by number of school days (60 vs 59)
ax.plot(gc_2023_avg.index, gc_2023_avg.values / 60, label='GC 23-24', marker='o', color='blue')
ax.plot(sv_2023_avg.index, sv_2023_avg.values / 60, label='SV 23-24', marker='o', color='green')
ax.plot(gc_2024_avg.index, gc_2024_avg.values / 59, label='GC 24-25', marker='s', color='red')
ax.plot(sv_2024_avg.index, sv_2024_avg.values / 59, label='SV 24-25', marker='s', color='orange')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Proportion of Average Absences', fontsize=14)
ax.set_title('Normalized Average Absences per Period (T2)', fontsize=16, pad=15)
ax.legend(loc='best')
ax.grid(True)

plt.tight_layout()
plt.show()

#----------------------------------------
# 6. Calculate Differences (SV - GC)
#----------------------------------------

diff_2023 = sv_2023_avg - gc_2023_avg
diff_2024 = sv_2024_avg - gc_2024_avg

print("\nDifference (SV - GC) 2023:\n", diff_2023)
print("Difference (SV - GC) 2024:\n", diff_2024)

#----------------------------------------
# 7. Plot Differences
#----------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.plot(diff_2023.index, diff_2023.values, label='Diff 23-24 (SV-GC)', marker='x', color='purple', linestyle='-')
ax.plot(diff_2024.index, diff_2024.values, label='Diff 24-25 (SV-GC)', marker='x', color='brown', linestyle='--')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Difference in Average Absences', fontsize=14)
ax.set_title('SV-GC Differences in Average Absences (T2)', fontsize=16, pad=15)
ax.legend(loc='best')
ax.grid(True)

plt.tight_layout()
plt.show()

#----------------------------------------
# 8. Difference-in-Difference (DID) Analysis
#----------------------------------------

# Prepare data for DID regression
gc_2023_long = pd.DataFrame(gc_2023_avg).reset_index()
gc_2023_long.columns = ['period', 'avg_absences']
gc_2023_long['school'] = 0
gc_2023_long['year'] = 0

sv_2023_long = pd.DataFrame(sv_2023_avg).reset_index()
sv_2023_long.columns = ['period', 'avg_absences']
sv_2023_long['school'] = 1
sv_2023_long['year'] = 0

gc_2024_long = pd.DataFrame(gc_2024_avg).reset_index()
gc_2024_long.columns = ['period', 'avg_absences']
gc_2024_long['school'] = 0
gc_2024_long['year'] = 1

sv_2024_long = pd.DataFrame(sv_2024_avg).reset_index()
sv_2024_long.columns = ['period', 'avg_absences']
sv_2024_long['school'] = 1
sv_2024_long['year'] = 1

# Combine into one DataFrame
df_reg = pd.concat([gc_2023_long, sv_2023_long, gc_2024_long, sv_2024_long])
df_reg['school_year'] = df_reg['school'] * df_reg['year']

#----------------------------------------
# 9. Run DID Regression
#----------------------------------------

# Regression via sklearn
lr = LinearRegression()
X = df_reg[['school', 'year', 'school_year']]
y = df_reg['avg_absences']
lr.fit(X, y)
print(f'\nSklearn Regression Coefficients: {lr.coef_}')

# Regression via statsmodels
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg).fit()
print(ols_model.summary())

#----------------------------------------
# 10. Calculate Manual DID
#----------------------------------------

mean_gc_2023 = gc_2023_long['avg_absences'].mean()
mean_sv_2023 = sv_2023_long['avg_absences'].mean()
mean_gc_2024 = gc_2024_long['avg_absences'].mean()
mean_sv_2024 = sv_2024_long['avg_absences'].mean()

gc_diff = mean_gc_2024 - mean_gc_2023
sv_diff = mean_sv_2024 - mean_sv_2023
did = sv_diff - gc_diff

print(f'\nManual DID Calculation:')
print(f'Mean GC Absences Before: {mean_gc_2023:.2f}')
print(f'Mean SV Absences Before: {mean_sv_2023:.2f}')
print(f'Mean GC Absences After: {mean_gc_2024:.2f}')
print(f'Mean SV Absences After: {mean_sv_2024:.2f}')
print(f'DID in Mean Absences: {did:.2f}')

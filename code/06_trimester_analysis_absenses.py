

import pandas as pd
import matplotlib.pyplot as plt

gc_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name= 'GC - Absence')
sv_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name= 'SV - Absence')
gc_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name= 'GC - Absences')
sv_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name= 'SV - Absences')
gc_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name= 'GC - Absence')
sv_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name= 'SV - Absence')
gc_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name= 'GC - Absences')
sv_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name= 'SV - Absences')
#columns for averages
columns = [1, 2, 3, 4, 5]

#averages for each df
gc_2023_avg_t1 = gc_2023_t1[columns].mean()
sv_2023_avg_t1 = sv_2023_t1[columns].mean()
gc_2024_avg_t1 = gc_2024_t1[columns].mean()
sv_2024_avg_t1 = sv_2024_t1[columns].mean()
gc_2023_avg_t2 = gc_2023_t2[columns].mean()
sv_2023_avg_t2 = sv_2023_t2[columns].mean()
gc_2024_avg_t2 = gc_2024_t2[columns].mean()
sv_2024_avg_t2 = sv_2024_t2[columns].mean()
#2023 T1 days:60
#2024 T1 days:59
#2023 T2 days:54
#2024 T2 days:59
print("GC 2023 Averages:\n", gc_2023_avg_t1)
print("SV 2023 Averages:\n", sv_2023_avg_t1)
print("GC 2024 Averages:\n", gc_2024_avg_t1)
print("SV 2024 Averages:\n", sv_2024_avg_t1)
print("GC 2023 Averages:\n", gc_2023_avg_t2)
print("SV 2023 Averages:\n", sv_2023_avg_t2)
print("GC 2024 Averages:\n", gc_2024_avg_t2)
print("SV 2024 Averages:\n", sv_2024_avg_t2)

#make the line chart
plt.figure(figsize=(10,6))

#each trimester with different colors
plt.plot(gc_2023_avg_t1.index, gc_2023_avg_t1.values, label='GC 23-24 T1', marker='o', color='blue')
plt.plot(sv_2023_avg_t1.index, sv_2023_avg_t1.values, label='SV 23-24 T1', marker='o', color='green')
plt.plot(gc_2024_avg_t1.index, gc_2024_avg_t1.values, label='GC 24-25 T1', marker='o', color='red')
plt.plot(sv_2024_avg_t1.index, sv_2024_avg_t1.values, label='SV 24-25 T1', marker='o', color='orange')

plt.plot(gc_2023_avg_t2.index, gc_2023_avg_t2.values, label='GC 23-24 T2', marker='o', color='pink')
plt.plot(sv_2023_avg_t2.index, sv_2023_avg_t2.values, label='SV 23-24 T2', marker='o', color='yellow')
plt.plot(gc_2024_avg_t2.index, gc_2024_avg_t2.values, label='GC 24-25 T2', marker='o', color='purple')
plt.plot(sv_2024_avg_t2.index, sv_2024_avg_t2.values, label='SV 24-25 T2', marker='o', color='black')
#labels and title
plt.xlabel('Period')
plt.ylabel('Average Absences')
plt.title('Average Absences Per Period')
plt.legend()

#make the line chart
plt.figure(figsize=(10,6))

#each trimester with different colors
plt.plot(gc_2023_avg_t1.index, (gc_2023_avg_t1.values)/60, label='GC 23-24', marker='o', color='blue')
plt.plot(sv_2023_avg_t1.index, (sv_2023_avg_t1.values)/60, label='SV 23-24', marker='o', color='green')
plt.plot(gc_2024_avg_t1.index, (gc_2024_avg_t1.values)/59, label='GC 24-25', marker='o', color='red')
plt.plot(sv_2024_avg_t1.index, (sv_2024_avg_t1.values)/59, label='SV 24-25', marker='o', color='orange')

plt.plot(gc_2023_avg_t2.index, (gc_2023_avg_t2.values)/54, label='GC 23-24', marker='o', color='pink')
plt.plot(sv_2023_avg_t2.index, (sv_2023_avg_t2.values)/54, label='SV 23-24', marker='o', color='yellow')
plt.plot(gc_2024_avg_t2.index, (gc_2024_avg_t2.values)/59, label='GC 24-25', marker='o', color='purple')
plt.plot(sv_2024_avg_t2.index, (sv_2024_avg_t2.values)/59, label='SV 24-25', marker='o', color='black')

#labels and title
plt.xlabel('Period')
plt.ylabel('Proportion of Average Absences')
plt.title('Average Absences Proportion Per Period')
plt.legend()

#show the plot
plt.tight_layout()
plt.show()

# Calculate the differences between SV and GC for each year
diff_2023_t1 = sv_2023_avg_t1 - gc_2023_avg_t1
diff_2024_t1 = sv_2024_avg_t1 - gc_2024_avg_t1
diff_2023_t2 = sv_2023_avg_t2 - gc_2023_avg_t2
diff_2024_t2 = sv_2024_avg_t2 - gc_2024_avg_t2
print("\nDifference (SV - GC) 2023 T1:\n", diff_2023_t1)
print("Difference (SV - GC) 2024 T1:\n", diff_2024_t1)
print("\nDifference (SV - GC) 2023 T2:\n", diff_2023_t2)
print("Difference (SV - GC) 2024 T2:\n", diff_2024_t2)

# Make the line chart
plt.figure(figsize=(12, 8))


# Plotting the differences
plt.plot(diff_2023_t1.index, diff_2023_t1.values, label='Diff 23-24 (SV-GC) T1', marker='x', color='purple', linestyle='--')
plt.plot(diff_2024_t1.index, diff_2024_t1.values, label='Diff 24-25 (SV-GC) T1', marker='x', color='brown', linestyle='--')
# Plotting the differences
plt.plot(diff_2023_t2.index, diff_2023_t2.values, label='Diff 23-24 (SV-GC) T2', marker='x', color='blue', linestyle='--')
plt.plot(diff_2024_t2.index, diff_2024_t2.values, label='Diff 24-25 (SV-GC) T2', marker='x', color='green', linestyle='--')
# Labels and title
plt.xlabel('Period')
plt.ylabel('Average Absences / Difference')
plt.title('Average Absences Per Period and SV-GC Differences')
plt.legend()
plt.grid(True)


# Show the plot
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

# Prepare data for DID analysis
# Assuming the 'columns' represent periods and we are comparing 2023 and 2024 data
# We will treat GC as control (0) and SV as treatment (1)

# Data before (2023)T1
gc_2023_long_t1 = pd.DataFrame(gc_2023_avg_t1).reset_index()
gc_2023_long_t1.columns = ['period', 'avg_absences']
gc_2023_long_t1['school'] = 0  # 0 for GC
gc_2023_long_t1['year'] = 0 # 0 for before

sv_2023_long_t1 = pd.DataFrame(sv_2023_avg_t1).reset_index()
sv_2023_long_t1.columns = ['period', 'avg_absences']
sv_2023_long_t1['school'] = 1  # 1 for SV
sv_2023_long_t1['year'] = 0 # 0 for before


# Data after (2024)T1
gc_2024_long_t1 = pd.DataFrame(gc_2024_avg_t1).reset_index()
gc_2024_long_t1.columns = ['period', 'avg_absences']
gc_2024_long_t1['school'] = 0  # 0 for GC
gc_2024_long_t1['year'] = 1 # 1 for after

sv_2024_long_t1 = pd.DataFrame(sv_2024_avg_t1).reset_index()
sv_2024_long_t1.columns = ['period', 'avg_absences']
sv_2024_long_t1['school'] = 1  # 1 for SV
sv_2024_long_t1['year'] = 1 # 1 for after


# Combine all data
df_reg_t1 = pd.concat([gc_2023_long_t1, sv_2023_long_t1, gc_2024_long_t1, sv_2024_long_t1])

# Data before (2023)T2
gc_2023_long_t2 = pd.DataFrame(gc_2023_avg_t2).reset_index()
gc_2023_long_t2.columns = ['period', 'avg_absences']
gc_2023_long_t2['school'] = 0  # 0 for GC
gc_2023_long_t2['year'] = 0 # 0 for before

sv_2023_long_t2 = pd.DataFrame(sv_2023_avg_t2).reset_index()
sv_2023_long_t2.columns = ['period', 'avg_absences']
sv_2023_long_t2['school'] = 1  # 1 for SV
sv_2023_long_t2['year'] = 0 # 0 for before


# Data after (2024)T2
gc_2024_long_t2 = pd.DataFrame(gc_2024_avg_t2).reset_index()
gc_2024_long_t2.columns = ['period', 'avg_absences']
gc_2024_long_t2['school'] = 0  # 0 for GC
gc_2024_long_t2['year'] = 1 # 1 for after

sv_2024_long_t2 = pd.DataFrame(sv_2024_avg_t2).reset_index()
sv_2024_long_t2.columns = ['period', 'avg_absences']
sv_2024_long_t2['school'] = 1  # 1 for SV
sv_2024_long_t2['year'] = 1 # 1 for after


# Combine all data
df_reg_t2 = pd.concat([gc_2023_long_t2, sv_2023_long_t2, gc_2024_long_t2, sv_2024_long_t2])

# Create interaction term
df_reg_t1['school_year'] = df_reg_t1['school'] * df_reg_t1['year']
df_reg_t2['school_year'] = df_reg_t2['school'] * df_reg_t2['year']

# Regression via sklearn T1
lr = LinearRegression()
X = df_reg_t1[['school', 'year', 'school_year']]
y = df_reg_t1['avg_absences']
lr.fit(X, y)
print(f'Coefficients: {lr.coef_}')

# Regression via statsmodels
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg_t1).fit()
print(ols_model.summary())

# Calculate DID manually
mean_gc_2023_t1 = gc_2023_long_t1['avg_absences'].mean()
mean_sv_2023_t1 = sv_2023_long_t1['avg_absences'].mean()
mean_gc_2024_t1 = gc_2024_long_t1['avg_absences'].mean()
mean_sv_2024_t1 = sv_2024_long_t1['avg_absences'].mean()

gc_diff_t1 = mean_gc_2024_t1 - mean_gc_2023_t1
sv_diff_t1 = mean_sv_2024_t1 - mean_sv_2023_t1
did_t1 = sv_diff_t1 - gc_diff_t1

print(f'T1 Mean GC absences before: {mean_gc_2023_t1:.2f}')
print(f'T1 Mean SV absences before: {mean_sv_2023_t1:.2f}')
print(f'T1 Mean GC absences after: {mean_gc_2024_t1:.2f}')
print(f'T1 Mean SV absences after: {mean_sv_2024_t1:.2f}')
print(f'T1 DID in mean absences: {did_t1:.2f}')

# Regression via sklearn T2
lr = LinearRegression()
X = df_reg_t2[['school', 'year', 'school_year']]
y = df_reg_t2['avg_absences']
lr.fit(X, y)
print(f'Coefficients: {lr.coef_}')

# Regression via statsmodels
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg_t2).fit()
print(ols_model.summary())

# Calculate DID manually
mean_gc_2023_t2 = gc_2023_long_t2['avg_absences'].mean()
mean_sv_2023_t2 = sv_2023_long_t2['avg_absences'].mean()
mean_gc_2024_t2 = gc_2024_long_t2['avg_absences'].mean()
mean_sv_2024_t2 = sv_2024_long_t2['avg_absences'].mean()

gc_diff_t2 = mean_gc_2024_t2 - mean_gc_2023_t2
sv_diff_t2 = mean_sv_2024_t2 - mean_sv_2023_t2
did_t2 = sv_diff_t2 - gc_diff_t2

print(f'T2 Mean GC absences before: {mean_gc_2023_t2:.2f}')
print(f'T2 Mean SV absences before: {mean_sv_2023_t2:.2f}')
print(f'T2 Mean GC absences after: {mean_gc_2024_t2:.2f}')
print(f'T2 Mean SV absences after: {mean_sv_2024_t2:.2f}')
print(f'T2 DID in mean absences: {did_t2:.2f}')

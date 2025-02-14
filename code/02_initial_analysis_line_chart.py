

import pandas as pd
import matplotlib.pyplot as plt

gc_2023 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name= 'GC - Absence')
sv_2023 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name= 'SV - Absence')
gc_2024 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name= 'GC - Absences')
sv_2024 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name= 'SV - Absences')

#columns for averages
columns = [1, 2, 3, 4, 5]

#averages for each df
gc_2023_avg = gc_2023[columns].mean()
sv_2023_avg = sv_2023[columns].mean()
gc_2024_avg = gc_2024[columns].mean()
sv_2024_avg = sv_2024[columns].mean()
#2023 T1 days:60
#2024 T1 days:59
print("GC 2023 Averages:\n", gc_2023_avg)
print("SV 2023 Averages:\n", sv_2023_avg)
print("GC 2024 Averages:\n", gc_2024_avg)
print("SV 2024 Averages:\n", sv_2024_avg)

#make the line chart
plt.figure(figsize=(10,6))

#each trimester with different colors
plt.plot(gc_2023_avg.index, gc_2023_avg.values, label='GC 23-24', marker='o', color='blue')
plt.plot(sv_2023_avg.index, sv_2023_avg.values, label='SV 23-24', marker='o', color='green')
plt.plot(gc_2024_avg.index, gc_2024_avg.values, label='GC 24-25', marker='o', color='red')
plt.plot(sv_2024_avg.index, sv_2024_avg.values, label='SV 24-25', marker='o', color='orange')

#labels and title
plt.xlabel('Period')
plt.ylabel('Average Absences')
plt.title('Average Absences Per Period')
plt.legend()

#make the line chart
plt.figure(figsize=(10,6))

#each trimester with different colors
plt.plot(gc_2023_avg.index, (gc_2023_avg.values)/60, label='GC 23-24', marker='o', color='blue')
plt.plot(sv_2023_avg.index, (sv_2023_avg.values)/60, label='SV 23-24', marker='o', color='green')
plt.plot(gc_2024_avg.index, (gc_2024_avg.values)/59, label='GC 24-25', marker='o', color='red')
plt.plot(sv_2024_avg.index, (sv_2024_avg.values)/59, label='SV 24-25', marker='o', color='orange')

#labels and title
plt.xlabel('Period')
plt.ylabel('Proportion of Average Absences')
plt.title('Average Absences Proportion Per Period')
plt.legend()

#show the plot
plt.tight_layout()
plt.show()

# Calculate the differences between SV and GC for each year
diff_2023 = sv_2023_avg - gc_2023_avg
diff_2024 = sv_2024_avg - gc_2024_avg

print("\nDifference (SV - GC) 2023:\n", diff_2023)
print("Difference (SV - GC) 2024:\n", diff_2024)

# Make the line chart
plt.figure(figsize=(12, 8))


# Plotting the differences
plt.plot(diff_2023.index, diff_2023.values, label='Diff 23-24 (SV-GC)', marker='x', color='purple', linestyle='--')
plt.plot(diff_2024.index, diff_2024.values, label='Diff 24-25 (SV-GC)', marker='x', color='brown', linestyle='--')

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

# Data before (2023)
gc_2023_long = pd.DataFrame(gc_2023_avg).reset_index()
gc_2023_long.columns = ['period', 'avg_absences']
gc_2023_long['school'] = 0  # 0 for GC
gc_2023_long['year'] = 0 # 0 for before

sv_2023_long = pd.DataFrame(sv_2023_avg).reset_index()
sv_2023_long.columns = ['period', 'avg_absences']
sv_2023_long['school'] = 1  # 1 for SV
sv_2023_long['year'] = 0 # 0 for before


# Data after (2024)
gc_2024_long = pd.DataFrame(gc_2024_avg).reset_index()
gc_2024_long.columns = ['period', 'avg_absences']
gc_2024_long['school'] = 0  # 0 for GC
gc_2024_long['year'] = 1 # 1 for after

sv_2024_long = pd.DataFrame(sv_2024_avg).reset_index()
sv_2024_long.columns = ['period', 'avg_absences']
sv_2024_long['school'] = 1  # 1 for SV
sv_2024_long['year'] = 1 # 1 for after


# Combine all data
df_reg = pd.concat([gc_2023_long, sv_2023_long, gc_2024_long, sv_2024_long])

# Create interaction term
df_reg['school_year'] = df_reg['school'] * df_reg['year']


# Regression via sklearn
lr = LinearRegression()
X = df_reg[['school', 'year', 'school_year']]
y = df_reg['avg_absences']
lr.fit(X, y)
print(f'Coefficients: {lr.coef_}')

# Regression via statsmodels
ols_model = ols('avg_absences ~ school + year + school_year', data=df_reg).fit()
print(ols_model.summary())

# Calculate DID manually
mean_gc_2023 = gc_2023_long['avg_absences'].mean()
mean_sv_2023 = sv_2023_long['avg_absences'].mean()
mean_gc_2024 = gc_2024_long['avg_absences'].mean()
mean_sv_2024 = sv_2024_long['avg_absences'].mean()

gc_diff = mean_gc_2024 - mean_gc_2023
sv_diff = mean_sv_2024 - mean_sv_2023
did = sv_diff - gc_diff

print(f'Mean GC absences before: {mean_gc_2023:.2f}')
print(f'Mean SV absences before: {mean_sv_2023:.2f}')
print(f'Mean GC absences after: {mean_gc_2024:.2f}')
print(f'Mean SV absences after: {mean_sv_2024:.2f}')
print(f'DID in mean absences: {did:.2f}')
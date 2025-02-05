import pandas as pd
import matplotlib.pyplot as plt

gc_2023 = pd.read_excel('../data/23-24 T1 Attendance.xlsx', sheet_name= 'GC - Absence')
sv_2023 = pd.read_excel('../data/23-24 T1 Attendance.xlsx', sheet_name= 'SV - Absence')
gc_2024 = pd.read_excel('../data/24-25 T1 Attendance.xlsx', sheet_name= 'GC - Absences')
sv_2024 = pd.read_excel('../data/24-25 T1 Attendance.xlsx', sheet_name= 'SV - Absences')

#columns for averages
columns = [1, 2, 3, 4, 5]

#averages for each df
gc_2023_avg = gc_2023[columns].mean()
sv_2023_avg = sv_2023[columns].mean()
gc_2024_avg = gc_2024[columns].mean()
sv_2024_avg = sv_2024[columns].mean()

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

#show the plot
plt.tight_layout()
plt.show()
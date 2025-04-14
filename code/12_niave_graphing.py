import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========= CONFIGURATION =========
# File paths and sheet names
file_info = [
    ("./data/23-24 T1 Attendance.xlsx", "GC - Absence", 2023, 1, "GC"),
    ("./data/23-24 T1 Attendance.xlsx", "SV - Absence", 2023, 1, "SV"),
    ("./data/23-24 T2 Attendance.xlsx", "GC - Absence", 2023, 2, "GC"),
    ("./data/23-24 T2 Attendance.xlsx", "SV - Absence", 2023, 2, "SV"),
    ("./data/24-25 T1 Attendance.xlsx", "GC - Absences", 2024, 1, "GC"),
    ("./data/24-25 T1 Attendance.xlsx", "SV - Absences", 2024, 1, "SV"),
    ("./data/24-25 T2 Attendance.xlsx", "GC - Absences", 2024, 2, "GC"),
    ("./data/24-25 T2 Attendance.xlsx", "SV - Absences", 2024, 2, "SV")
]


# Path to demographics file (optional — used for merging if available)
demographics_path = "./data/Attendance Demographics 2024-2025.csv"

# ========= LOAD AND COMBINE DATA =========

frames = []
try:
    demographics = pd.read_csv(demographics_path, dtype=str)
    demographics.rename(columns={"Student Number": "Number"}, inplace=True)
except FileNotFoundError:
    demographics = None
    print("⚠️ Demographics file not found. Proceeding without it.")

for path, sheet, year, trimester, school in file_info:
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    df["Number"] = df["Number"].astype(str)
    if demographics is not None:
        df = df.merge(demographics, on="Number", how="left")
    df["Year"] = year
    df["Trimester"] = trimester
    df["School"] = school
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
combined["Total"] = pd.to_numeric(combined["Total"], errors="coerce")
combined["Trimester"] = pd.to_numeric(combined["Trimester"], errors="coerce")

# Add a row for the missing trimester 3
missing_row = pd.DataFrame([{"Time Period": "2023 T3", "School": "GC", "Total": None},
                            {"Time Period": "2023 T3", "School": "SV", "Total": None}])



# ========= AGGREGATE: Average Absences =========
combined["Time Period"] = combined.apply(lambda row: f"{row['Year']} T{row['Trimester']}", axis=1)

avg_summary = (
    combined.groupby(["Time Period", "School"])["Total"]
    .mean()
    .reset_index().sort_values(["Time Period", "School"])
)
avg_summary = pd.concat([avg_summary, missing_row], ignore_index=True).sort_values(["Time Period", "School"])



# ========= PLOT: Raw Averages (Bar Chart) =========
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_summary, x="Time Period", y="Total", hue="School", palette={"GC": "green", "SV": "blue"})
plt.title("Raw Average Absences by School and Time Period")
plt.ylabel("Average Absences")
plt.xlabel("Time Period")
plt.grid(True)
plt.legend(title="School")
plt.show()

# ========= CALCULATE: Raw Differences Over Time =========
diff_summary = pd.DataFrame()
for trimester in [1, 2]:
    # Filter data for the current trimester
    trim_data = avg_summary[avg_summary["Time Period"].str.contains(f"T{trimester}")]

    # Pivot the table to have schools as columns and years as index
    trim_pivot = trim_data.pivot(index="Time Period", columns="School", values="Total")

    # Calculate the difference between SV and GC for each year
    trim_pivot["Difference (SV-GC)"] = trim_pivot["SV"] - trim_pivot["GC"]

    # Reset index to make 'Year' and 'Trimester' regular columns
    trim_pivot = trim_pivot.reset_index()

    # Add a 'Trimester' column
    trim_pivot["Trimester"] = f"T{trimester}"

    # Append to the overall summary
    diff_summary = pd.concat([diff_summary, trim_pivot], ignore_index=True)

# ========= PLOT: Raw Differences (Bar Chart) =========
plt.figure(figsize=(10, 6))
sns.barplot(data=diff_summary, x="Time Period", y="Difference (SV-GC)", palette=["purple"])
plt.title("Raw Difference in Average Absences (SV - GC) by Trimester")
plt.ylabel("Difference (SV - GC)")
plt.xlabel("Time Period")
plt.grid(True)
plt.show()

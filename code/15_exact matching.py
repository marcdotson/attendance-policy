# exact_matching_analysis.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# =====================
# STEP 1: Load and Preprocess Data
# =====================
print("Loading and preparing data...")

# File paths
demographics_file = "data/Attendance Demographics 2024-2025.csv"
attendance_2023_gc = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="GC - Absence", dtype=str)
attendance_2023_sv = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="SV - Absence", dtype=str)
attendance_2024_gc = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="GC - Absences", dtype=str)
attendance_2024_sv = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="SV - Absences", dtype=str)

# Add school labels
attendance_2023_gc["School"] = 0
attendance_2023_sv["School"] = 1
attendance_2024_gc["School"] = 0
attendance_2024_sv["School"] = 1

# Combine and clean
df_2023 = pd.concat([attendance_2023_gc, attendance_2023_sv], ignore_index=True)
df_2024 = pd.concat([attendance_2024_gc, attendance_2024_sv], ignore_index=True)
df_2023.columns = df_2023.columns.str.strip()
df_2024.columns = df_2024.columns.str.strip()

# Convert ID columns to string
df_2023["Number"] = df_2023["Number"].astype(str)
df_2024["Number"] = df_2024["Number"].astype(str)

# Load demographics and convert ID to string
demographics_df = pd.read_csv(demographics_file)
demographics_df["Student Number"] = demographics_df["Student Number"].astype(str)

# Merge with demographics
df_2023 = df_2023.merge(demographics_df, left_on="Number", right_on="Student Number", how="left")
df_2024 = df_2024.merge(demographics_df, left_on="Number", right_on="Student Number", how="left")

# Re-assign School column after merge
school_map_2023 = pd.concat([attendance_2023_gc, attendance_2023_sv]).set_index("Number")["School"].to_dict()
school_map_2024 = pd.concat([attendance_2024_gc, attendance_2024_sv]).set_index("Number")["School"].to_dict()
df_2023["School"] = df_2023["Number"].map(school_map_2023)
df_2024["School"] = df_2024["Number"].map(school_map_2024)

# =====================
# STEP 2: Exact Matching
# =====================
print("Performing exact matching...")

# Preprocessing categorical variables for exact matching
df_2023["Gender_bin"] = df_2023["Gender"].map({"M": 0, "F": 1}).fillna(0)
df_2023["ESL_bin"] = df_2023["ESL"].apply(lambda x: 1 if str(x).upper() == "Y" else 0)
df_2023["Grade Level"] = pd.to_numeric(df_2023["Grade Level"], errors="coerce").fillna(9)
df_2023["Hispanic_bin"] = df_2023["Hispanic"].apply(lambda x: 1 if str(x).upper() == "Y" else 0)

# Define matching variables
matching_vars = ["Gender_bin", "Grade Level", "ESL_bin", "Hispanic_bin"]

# Split treatment and control
treated = df_2023[df_2023["School"] == 1]
control = df_2023[df_2023["School"] == 0]

# Perform exact matching
# Perform exact matching
matched_rows = []
for _, treated_row in treated.iterrows():
    condition = (control[matching_vars] == treated_row[matching_vars].values).all(axis=1)
    potential_matches = control[condition]
    if not potential_matches.empty:
        match = potential_matches.sample(n=1, random_state=42)
        matched_rows.append(treated_row.to_dict())
        matched_rows.append(match.iloc[0].to_dict())  # Safely extract first row

matched_data = pd.DataFrame(matched_rows).reset_index(drop=True)
print(f"Matched sample size: {len(matched_data)} students")

# =====================
# STEP 3: Pre/Post Absence Comparison
# =====================
print("Analyzing absence changes...")

df_2023["Total"] = pd.to_numeric(df_2023["Total"], errors="coerce")
df_2024["Total"] = pd.to_numeric(df_2024["Total"], errors="coerce")

matched_ids = matched_data["Number"].unique()
matched_pre = df_2023[df_2023["Number"].isin(matched_ids)][["Number", "School", "Total"]]
matched_post = df_2024[df_2024["Number"].isin(matched_ids)][["Number", "Total"]]

matched_df = matched_pre.rename(columns={"Total": "PrePolicyAbsences"}).merge(
    matched_post.rename(columns={"Total": "PostPolicyAbsences"}), on="Number"
)

matched_df.dropna(inplace=True)
matched_df["ChangeInAbsences"] = matched_df["PostPolicyAbsences"] - matched_df["PrePolicyAbsences"]

# =====================
# STEP 4: T-test
# =====================
print("Running t-test on matched change scores...")

control_change = matched_df[matched_df["School"] == 0]["ChangeInAbsences"]
treatment_change = matched_df[matched_df["School"] == 1]["ChangeInAbsences"]

t_stat, p_val = ttest_ind(treatment_change, control_change, equal_var=False)
print(f"Exact Matched Change t-test: t={t_stat:.2f}, p={p_val:.4f}")

# =====================
# STEP 5: Visualization
# =====================


# Group means
group_means = matched_df.groupby("School")["ChangeInAbsences"].mean()
labels = ["Green Canyon (Control)", "Sky View (Policy)"]
colors = ["#66bb6a", "#42a5f5"]

# Create bar chart
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, group_means, color=colors, width=0.6)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.3,
            f"{yval:.2f} days", ha='center', va='bottom', fontsize=12, weight='bold')

# Style and spacing
ax.set_title("Change in Absences After Policy", fontsize=12, pad=10)
ax.set_ylabel("Avg. Change in Absences", fontsize=12, labelpad=10)
ax.set_ylim(0, max(group_means) + 2)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()






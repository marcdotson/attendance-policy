import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind

# Load the demographics data
demographics_file = r"data\Attendance Demographics 2024-2025.csv"
demographics_df = pd.read_csv(demographics_file)

# Load attendance data (replace with actual file paths)
attendance_2023_gc = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="GC - Absence", dtype=str)
attendance_2023_sv = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="SV - Absence", dtype=str)
attendance_2024_gc = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="GC - Absences", dtype=str)
attendance_2024_sv = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="SV - Absences", dtype=str)
# Convert column names to string to avoid NaN during concat
attendance_2023_gc.columns = attendance_2023_gc.columns.map(str)
attendance_2023_sv.columns = attendance_2023_sv.columns.map(str)
attendance_2024_gc.columns = attendance_2024_gc.columns.map(str)
attendance_2024_sv.columns = attendance_2024_sv.columns.map(str)

print("Attendance 2023 GC Columns:", attendance_2023_gc.columns.tolist())
# Concatenate attendance data and add school labels
attendance_2023_gc["School"] = 0  # GC (Control)
attendance_2023_sv["School"] = 1  # SV (Treatment)
attendance_2024_gc["School"] = 0
attendance_2024_sv["School"] = 1

df_2023 = pd.concat([attendance_2023_gc, attendance_2023_sv], ignore_index=True)
df_2024 = pd.concat([attendance_2024_gc, attendance_2024_sv], ignore_index=True)
df_2023.columns = df_2023.columns.str.strip()
df_2024.columns = df_2024.columns.str.strip()


print("df_2023 Columns:", df_2023.columns.tolist())
print("df_2024 Columns:", df_2024.columns.tolist())
print(df_2023.columns)

# Merge attendance with demographics on Student ID
df_2023["Number"] = df_2023["Number"].astype(str)
df_2024["Number"] = df_2024["Number"].astype(str)
demographics_df["Student Number"] = demographics_df["Student Number"].astype(str)
df_2023.rename(columns={"School": "School_from_attendance"}, inplace=True)
df_2024.rename(columns={"School": "School_from_attendance"}, inplace=True)
df_2023 = df_2023.merge(demographics_df, left_on="Number", right_on="Student Number")
df_2024 = df_2024.merge(demographics_df, left_on="Number", right_on="Student Number")

# Define matching variables
df_2023["Gender"] = df_2023["Gender"].map({"M": 0, "F": 1})
df_2023["ESL"] = df_2023["ESL"].map({"Y": 1, "N": 0})
df_2023["SWD"] = df_2023["SWD"].map({"Y": 1, "N": 0})
df_2023["PrePolicyAbsences"] = df_2023["Total"]

# Fit logistic regression for propensity scores

X = df_2023[[
    "Gender", "ESL", "Grade Level", "Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian", "PrePolicyAbsences"
]].copy()

y = df_2023["School_from_attendance"]

# Gender: M/F → 0/1
X["Gender"] = pd.to_numeric(X["Gender"], errors="coerce").fillna(0).astype(int)

# ESL: Y → 1, else 0
X["ESL"] = X["ESL"].apply(lambda x: 1 if str(x).strip().upper() == "Y" else 0)

# Grade Level: numeric
X["Grade Level"] = pd.to_numeric(X["Grade Level"], errors="coerce")
X["Grade Level"] = X["Grade Level"].fillna(X["Grade Level"].median())

# Race/Ethnicity: Y → 1, else 0
race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]
for col in race_cols:
    X[col] = X[col].apply(lambda x: 1 if str(x).strip().upper() == "Y" else 0)

# Pre-policy absences
X["PrePolicyAbsences"] = pd.to_numeric(X["PrePolicyAbsences"], errors="coerce")
X["PrePolicyAbsences"] = X["PrePolicyAbsences"].fillna(X["PrePolicyAbsences"].median())

# Fit model
logit = LogisticRegression()
df_2023["PropensityScore"] = logit.fit(X, y).predict_proba(X)[:, 1]


# Perform nearest neighbor matching
treated = df_2023[df_2023["School_from_attendance"] == 1]
control = df_2023[df_2023["School_from_attendance"] == 0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[["PropensityScore"]])
distances, indices = nn.kneighbors(treated[["PropensityScore"]])
matched_control = control.iloc[indices.flatten()]
matched_data = pd.concat([treated, matched_control])

# Compare post-policy absences
df_2024_matched = df_2024[df_2024["Number"].isin(matched_data["Number"])].copy()  # <-- important!
df_2024_matched["Total"] = pd.to_numeric(df_2024_matched["Total"], errors="coerce")
df_2024_matched = df_2024_matched.dropna(subset=["Total"])


# Run t-test
t_stat, p_value = ttest_ind(
    df_2024_matched[df_2024_matched["School_from_attendance"] == 1]["Total"],
    df_2024_matched[df_2024_matched["School_from_attendance"] == 0]["Total"],
    equal_var=False
)

print(f"T-statistic: {t_stat}, P-value: {p_value}")

import matplotlib.pyplot as plt

# Group means
group_means = df_2024_matched.groupby("School_from_attendance")["Total"].mean()
group_labels = ["GC", "SV"]
colors = ["green", "blue"]

# Bar chart
plt.figure(figsize=(8, 6))
plt.bar(group_labels, group_means, color=colors)
plt.title("Average Post-Policy Absences (Matched Students)",)
plt.ylabel("Average Absences")
plt.ylim(0, max(group_means) + 1)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()


import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(x="School_from_attendance", y="Total", data=df_2024_matched, palette=colors)
plt.xticks([0, 1], ["GC", "SV"])  # Set labels for the x-axis
plt.title("Distribution of Post-Policy Absences (Matched Students)")
plt.ylabel("Absences")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.show()
matched_pre = df_2023[df_2023["Number"].isin(matched_data["Number"])].copy()
matched_pre["PrePolicyAbsences"] = pd.to_numeric(matched_pre["PrePolicyAbsences"], errors="coerce")
matched_pre_group = matched_pre.groupby("School_from_attendance")["PrePolicyAbsences"].mean()
matched_post_group = df_2024_matched.groupby("School_from_attendance")["Total"].mean()

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure numeric
df_2024_matched["Grade Level"] = pd.to_numeric(df_2024_matched["Grade Level"], errors="coerce")
df_2024_matched["Total"] = pd.to_numeric(df_2024_matched["Total"], errors="coerce")

# Group by grade and school
grade_grouped = df_2024_matched[df_2024_matched["Grade Level"].isin([10, 11, 12])].groupby(["Grade Level", "School_from_attendance"])["Total"].mean().unstack().fillna(0)

# Plot
ax = grade_grouped.plot(kind="bar", figsize=(10, 6), color = ["green", "blue"])
plt.title("Average Absences by Grade Level",)
plt.ylabel("Average Absences")
plt.xlabel("Grade Level")

# Add labels to bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 5), textcoords='offset points')


plt.xticks(rotation=0, ticks=[0,1,2], labels=["10th", "11th", "12th"])
plt.legend(["GC", "SV"], title="School", loc='upper right')
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Convert race columns from 'Y'/'N' to 1/0
race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]
for col in race_cols:
    df_2024_matched[col] = df_2024_matched[col].apply(lambda x: 1 if str(x).strip().upper() == "Y" else 0)

# Build race average dataframe
race_avgs = []

for race in race_cols:
    gc_vals = df_2024_matched[(df_2024_matched[race] == 1) & (df_2024_matched["School_from_attendance"] == 0)]["Total"]
    sv_vals = df_2024_matched[(df_2024_matched[race] == 1) & (df_2024_matched["School_from_attendance"] == 1)]["Total"]

    gc_mean = gc_vals.mean()
    sv_mean = sv_vals.mean()
    count = df_2024_matched[df_2024_matched[race] == 1].shape[0]

    race_avgs.append({
        "Race": race,
        "GC": gc_mean,
        "SV": sv_mean,
        "Count": count
    })

# Create race_df
race_df = pd.DataFrame(race_avgs).fillna(0)

# Optional: filter races with at least 5 students (adjustable)
race_df = race_df[race_df["Count"] >= 5]

# Melt for plotting
race_melted = race_df.melt(id_vars="Race", value_vars=["GC", "SV"],
                           var_name="School", value_name="Average Absences") 

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Race", y="Average Absences", hue="School", data=race_melted, palette=colors, ax=ax)
plt.title("Average Absences by Race (Students with N ≥ 5)")
plt.xticks(rotation=45)
plt.ylabel("Average Absences")
plt.xlabel("Race")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.ylim(0, max(race_melted["Average Absences"]) + 1)
# Add labels to bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 5), textcoords='offset points')

# Move legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()

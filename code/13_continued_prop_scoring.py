#----------------------------------------
# Purpose: Propensity Score Matching + Change-in-Absences Analysis
#----------------------------------------
#
# This script matches similar students between GC (control) and SV (treatment) 
# based on demographics and pre-policy attendance, then compares the *change in absences* 
# (after - before) between schools.
#
# This focuses on how much students' absences changed, NOT just final absence levels.
#
#----------------------------------------

# =====================
# STEP 1: Load and Preprocess Data
# =====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind

print("Loading data...")

# Load datasets
demographics_file = "data/Attendance Demographics 2024-2025.csv"
demographics_df = pd.read_csv(demographics_file)

attendance_2023_gc = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="GC - Absence", dtype=str)
attendance_2023_sv = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="SV - Absence", dtype=str)
attendance_2024_gc = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="GC - Absences", dtype=str)
attendance_2024_sv = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="SV - Absences", dtype=str)

# Label school
attendance_2023_gc["School"] = 0
attendance_2023_sv["School"] = 1
attendance_2024_gc["School"] = 0
attendance_2024_sv["School"] = 1

# Combine
df_2023 = pd.concat([attendance_2023_gc, attendance_2023_sv], ignore_index=True)
df_2024 = pd.concat([attendance_2024_gc, attendance_2024_sv], ignore_index=True)

# Fix columns
for df in [df_2023, df_2024]:
    df.columns = df.columns.str.strip()
    df["Number"] = df["Number"].astype(str)

# Save school mapping BEFORE merge
school_map = df_2023.set_index("Number")["School"].to_dict()

# Merge demographics
demographics_df["Student Number"] = demographics_df["Student Number"].astype(str)
df_2023 = df_2023.merge(demographics_df, left_on="Number", right_on="Student Number", how="left")
df_2024 = df_2024.merge(demographics_df, left_on="Number", right_on="Student Number", how="left")

# Ensure 'School' column present after merge
if 'School' not in df_2023.columns:
    print("[Warning] 'School' column missing after merge in df_2023. Reconstructing...")
    df_2023["School"] = df_2023["Number"].map(school_map)
if 'School' not in df_2024.columns:
    print("[Warning] 'School' column missing after merge in df_2024. Reconstructing...")
    df_2024["School"] = df_2024["Number"].map(school_map)

# =====================
# STEP 2: Propensity Score Matching
# =====================

print("Calculating propensity scores...")

race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]

X = df_2023[["Gender", "ESL", "Grade Level", "Total"] + race_cols].copy()

X["Gender"] = X["Gender"].map({"M": 0, "F": 1}).fillna(0)
X["ESL"] = X["ESL"].apply(lambda x: 1 if str(x).upper() == "Y" else 0)
X["Grade Level"] = pd.to_numeric(X["Grade Level"], errors="coerce").fillna(9)
X["Total"] = pd.to_numeric(X["Total"], errors="coerce")
X["Total"] = X["Total"].fillna(X["Total"].median())

for col in race_cols:
    X[col] = X[col].apply(lambda x: 1 if str(x).upper() == "Y" else 0)

# Fit logistic model
logit = LogisticRegression()
y = df_2023["School"]
df_2023["PropensityScore"] = logit.fit(X, y).predict_proba(X)[:, 1]

# Matching
print("Performing nearest neighbor matching...")

treated = df_2023[df_2023["School"] == 1]
control = df_2023[df_2023["School"] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["PropensityScore"]])
_, indices = nn.kneighbors(treated[["PropensityScore"]])

matched_control = control.iloc[indices.flatten()]
matched_data = pd.concat([treated, matched_control])

# =====================
# STEP 3: Change in Absences Calculation
# =====================

print("Calculating changes in absences...")

# Get pre- and post-policy data
matched_pre = df_2023[df_2023["Number"].isin(matched_data["Number"])]
matched_post = df_2024[df_2024["Number"].isin(matched_data["Number"])]

# Build final matched dataframe
matched_df = matched_pre[["Number", "School", "Total"] + race_cols + ["Gender", "ESL"]].rename(columns={"Total": "PrePolicyAbsences"})
matched_df = matched_df.merge(
    matched_post[["Number", "Total"]].rename(columns={"Total": "PostPolicyAbsences"}),
    on="Number"
)

matched_df["PrePolicyAbsences"] = pd.to_numeric(matched_df["PrePolicyAbsences"], errors="coerce")
matched_df["PostPolicyAbsences"] = pd.to_numeric(matched_df["PostPolicyAbsences"], errors="coerce")
matched_df.dropna(inplace=True)
matched_df["ChangeInAbsences"] = matched_df["PostPolicyAbsences"] - matched_df["PrePolicyAbsences"]

# Compare unmatched
df_2023["Total"] = pd.to_numeric(df_2023["Total"], errors="coerce")
df_2024["Total"] = pd.to_numeric(df_2024["Total"], errors="coerce")
unmatched_diff = df_2024.groupby("School")["Total"].mean() - df_2023.groupby("School")["Total"].mean()

# =====================
# STEP 4: Statistical Testing
# =====================

print("Running t-tests...")

# Matched t-test
control_change = matched_df[matched_df["School"] == 0]["ChangeInAbsences"]
treatment_change = matched_df[matched_df["School"] == 1]["ChangeInAbsences"]
t_stat, p_val = ttest_ind(treatment_change, control_change, equal_var=False)

print(f"Matched Change t-test: t={t_stat:.2f}, p={p_val:.4f}")

if p_val < 0.05:
    print("✅ Statistically significant difference detected.")
else:
    print("❌ No statistically significant difference detected.")

# =====================
# STEP 5: Visualizations
# =====================




# Propensity Score KDE Plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data=df_2023[df_2023["School"] == 0], x="PropensityScore", label="GC (Control)", fill=True, alpha=0.5, color="green")
sns.kdeplot(data=df_2023[df_2023["School"] == 1], x="PropensityScore", label="SV (Policy)", fill=True, alpha=0.5, color="blue")
plt.title("Propensity Score Distributions Before Matching")
plt.xlabel("Propensity Score")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

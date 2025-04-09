import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# STEP 1: Load and Combine T1 + T2 Data
# =====================
print("Loading data...")
demographics = pd.read_csv("data/Attendance Demographics 2024-2025.csv")
demographics["Student Number"] = demographics["Student Number"].astype(str)

def load_attendance(path, sheet, year, trimester, school):
    df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    df["Year"] = year
    df["Trimester"] = trimester
    df["School"] = school
    return df

files = [
    ("data/23-24 T1 Attendance.xlsx", "GC - Absence", 2023, 1, 0),
    ("data/23-24 T1 Attendance.xlsx", "SV - Absence", 2023, 1, 1),
    ("data/23-24 T2 Attendance.xlsx", "GC - Absence", 2023, 2, 0),
    ("data/23-24 T2 Attendance.xlsx", "SV - Absence", 2023, 2, 1),
    ("data/24-25 T1 Attendance.xlsx", "GC - Absences", 2024, 1, 0),
    ("data/24-25 T1 Attendance.xlsx", "SV - Absences", 2024, 1, 1),
    ("data/24-25 T2 Attendance.xlsx", "GC - Absences", 2024, 2, 0),
    ("data/24-25 T2 Attendance.xlsx", "SV - Absences", 2024, 2, 1),
]

dfs = [load_attendance(*args) for args in files]
df_all = pd.concat(dfs, ignore_index=True)
df_all.columns = df_all.columns.str.strip()
df_all["Number"] = df_all["Number"].astype(str)

# Merge with demographics
df_all = df_all.merge(demographics, left_on="Number", right_on="Student Number", how="left")

# Restore School column if dropped during merge
if "School" not in df_all.columns:
    print("Restoring School column from original files...")
    school_lookup = pd.concat(dfs, ignore_index=True).drop_duplicates("Number")[["Number", "School"]]
    df_all = df_all.merge(school_lookup, on="Number", how="left")

# =====================
# STEP 2: Aggregate Pre/Post Absences
# =====================
df_all["Total"] = pd.to_numeric(df_all["Total"], errors="coerce")

pre_df = df_all[df_all["Year"] == 2023]
post_df = df_all[df_all["Year"] == 2024]

pre_agg = pre_df.groupby("Number")["Total"].mean().rename("PrePolicyAbsences")
post_agg = post_df.groupby("Number")["Total"].mean().rename("PostPolicyAbsences")

race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]
latest_2023 = df_all[df_all["Year"] == 2023].sort_values(["Number", "Trimester"], ascending=False)
match_base = latest_2023.drop_duplicates("Number").copy()

match_base = match_base[["Number", "Gender", "ESL", "Grade Level", "School"] + race_cols]
match_base["Gender"] = match_base["Gender"].map({"M": 0, "F": 1}).fillna(0)
match_base["ESL"] = match_base["ESL"].apply(lambda x: 1 if str(x).upper() == "Y" else 0)
match_base["Grade Level"] = pd.to_numeric(match_base["Grade Level"], errors="coerce").fillna(9)

for col in race_cols:
    match_base[col] = match_base[col].apply(lambda x: 1 if str(x).upper() == "Y" else 0)

match_base = match_base.merge(pre_agg, on="Number").merge(post_agg, on="Number")
match_base.dropna(subset=["PrePolicyAbsences", "PostPolicyAbsences"], inplace=True)

# =====================
# STEP 3: Propensity Score Matching (with scaling)
# =====================
print("Calculating propensity scores...")

X = match_base[["Gender", "ESL", "Grade Level"] + race_cols + ["PrePolicyAbsences"]]
y = match_base["School"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logit = LogisticRegression(max_iter=500)
match_base["PropensityScore"] = logit.fit(X_scaled, y).predict_proba(X_scaled)[:, 1]

treated = match_base[match_base["School"] == 1]
control = match_base[match_base["School"] == 0]
nn = NearestNeighbors(n_neighbors=1).fit(control[["PropensityScore"]])
_, indices = nn.kneighbors(treated[["PropensityScore"]])
matched_control = control.iloc[indices.flatten()]
matched_df = pd.concat([treated, matched_control])
matched_df["ChangeInAbsences"] = matched_df["PostPolicyAbsences"] - matched_df["PrePolicyAbsences"]

# =====================
# STEP 4: Analysis & Visualization
# =====================
print("Analyzing matched difference in absences...")
control_change = matched_df[matched_df["School"] == 0]["ChangeInAbsences"]
treatment_change = matched_df[matched_df["School"] == 1]["ChangeInAbsences"]
t_stat, p_val = ttest_ind(treatment_change, control_change, equal_var=False)

print(f"\nMatched Change t-test: t={t_stat:.2f}, p={p_val:.4f}")
if p_val < 0.05:
    print("✅ Significant difference. The policy MAY have had an effect.")
else:
    print("⚠️ No significant difference. The policy likely had no measurable impact.")

# Bar chart with labels
bar_df = matched_df.groupby("School")["ChangeInAbsences"].mean().reset_index()
bar_df["School"] = bar_df["School"].map({0: "GC (Control)", 1: "SV (Policy)"})

plt.figure(figsize=(7, 5))
sns.barplot(data=bar_df, x="School", y="ChangeInAbsences", palette=["#4C72B0", "#DD8452"])
plt.title("Change in Absences After Policy (Matched Students)")
plt.ylabel("Absence Change (Post - Pre)")
plt.xlabel("")
for i, v in enumerate(bar_df["ChangeInAbsences"]):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Line chart for trends
print("Plotting trimester trend over time...")
plot_trend = df_all[df_all["Number"].isin(matched_df["Number"])].copy()
plot_trend["Total"] = pd.to_numeric(plot_trend["Total"], errors="coerce")

plot_df = plot_trend.groupby(["Year", "Trimester", "School"])["Total"].mean().reset_index()
plot_df["School"] = plot_df["School"].map({0: "GC (Control)", 1: "SV (Policy)"})


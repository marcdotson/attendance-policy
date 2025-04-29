#Continuation on prop scoring, just removing duplicates didn't change too much 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
 # Prevents rendering in interactive window

# =====================
# STEP 1: Load and Preprocess Data
# =====================
print("Loading data...")
demographics_file = "data/Attendance Demographics 2024-2025.csv"
demographics_df = pd.read_csv(demographics_file)

attendance_2023_gc = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="GC - Absence", dtype=str)
attendance_2023_sv = pd.read_excel("data/23-24 T2 Attendance.xlsx", sheet_name="SV - Absence", dtype=str)
attendance_2024_gc = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="GC - Absences", dtype=str)
attendance_2024_sv = pd.read_excel("data/24-25 T2 Attendance.xlsx", sheet_name="SV - Absences", dtype=str)

attendance_2023_gc["School"] = 0
attendance_2023_sv["School"] = 1
attendance_2024_gc["School"] = 0
attendance_2024_sv["School"] = 1

df_2023 = pd.concat([attendance_2023_gc, attendance_2023_sv], ignore_index=True)
df_2024 = pd.concat([attendance_2024_gc, attendance_2024_sv], ignore_index=True)

df_2023.columns = df_2023.columns.str.strip()
df_2024.columns = df_2024.columns.str.strip()
demographics_df["Student Number"] = demographics_df["Student Number"].astype(str)
df_2023["Number"] = df_2023["Number"].astype(str)
df_2024["Number"] = df_2024["Number"].astype(str)

school_map = df_2023.set_index("Number")["School"].to_dict()
df_2023 = df_2023.merge(demographics_df, left_on="Number", right_on="Student Number")
df_2024 = df_2024.merge(demographics_df, left_on="Number", right_on="Student Number")

if "School" not in df_2024.columns:
    df_2024["School"] = df_2024["Number"].map(school_map)
if "School" not in df_2023.columns:
    df_2023["School"] = df_2023["Number"].map(school_map)

# =====================
# STEP 2: Propensity Score Matching
# =====================
print("Calculating propensity scores...")

X = df_2023[[
    "Gender", "ESL", "Grade Level", "Hispanic", "White", "Black",
    "Asian", "Pacific Islander", "American Indian", "Total"
]].copy()
X["Gender"] = X["Gender"].map({"M": 0, "F": 1}).fillna(0)
X["ESL"] = X["ESL"].apply(lambda x: 1 if str(x).upper() == "Y" else 0)
X["Grade Level"] = pd.to_numeric(X["Grade Level"], errors="coerce").fillna(9)
X["Total"] = pd.to_numeric(X["Total"], errors="coerce")
X["Total"] = X["Total"].fillna(X["Total"].median())
race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]
for col in race_cols:
    X[col] = X[col].apply(lambda x: 1 if str(x).upper() == "Y" else 0)

logit = LogisticRegression()
y = df_2023["School"]
df_2023["PropensityScore"] = logit.fit(X, y).predict_proba(X)[:, 1]

print("Performing nearest neighbor matching...")
treated = df_2023[df_2023["School"] == 1]
control = df_2023[df_2023["School"] == 0]
nn = NearestNeighbors(n_neighbors=1).fit(control[["PropensityScore"]])
_, indices = nn.kneighbors(treated[["PropensityScore"]])
matched_control = control.iloc[indices.flatten()]
matched_data = pd.concat([treated, matched_control])

# ✅ ADD THIS BLOCK HERE to check for reused students
reused_counts = matched_control["Number"].value_counts()
reused_students = reused_counts[reused_counts > 1]

# Remove reused control students from matched data
if len(reused_students) > 0:
    print(f"\nNumber of reused control students: {len(reused_students)}")
    print("\nList of reused control student IDs:")
    print(reused_students)

# Filter out the reused control students
unique_control = matched_control[~matched_control["Number"].isin(reused_students.index)]
matched_data = pd.concat([treated, unique_control])

# ✅ ADD THIS BLOCK HERE to check for reused students
reused_counts = matched_control["Number"].value_counts()
reused_students = reused_counts[reused_counts > 1]

# Remove reused control students from matched data
if len(reused_students) > 0:
    print(f"\nNumber of reused control students: {len(reused_students)}")
    print("\nList of reused control student IDs:")
    print(reused_students)

# =====================
# STEP 3: Matched vs Unmatched Comparison
# =====================
print("Analyzing matched vs unmatched differences...")

matched_pre = df_2023[df_2023["Number"].isin(matched_data["Number"])]
matched_post = df_2024[df_2024["Number"].isin(matched_data["Number"])]

matched_df = matched_pre[["Number", "School", "Total"] + race_cols + ["Gender", "ESL"]].rename(columns={"Total": "PrePolicyAbsences"})
matched_df = matched_df.merge(
    matched_post[["Number", "Total"]].rename(columns={"Total": "PostPolicyAbsences"}),
    on="Number"
)
matched_df["PrePolicyAbsences"] = pd.to_numeric(matched_df["PrePolicyAbsences"], errors="coerce")
matched_df["PostPolicyAbsences"] = pd.to_numeric(matched_df["PostPolicyAbsences"], errors="coerce")
matched_df.dropna(inplace=True)
matched_df["ChangeInAbsences"] = matched_df["PostPolicyAbsences"] - matched_df["PrePolicyAbsences"]

# Unmatched group change
df_2023["Total"] = pd.to_numeric(df_2023["Total"], errors="coerce")
df_2024["Total"] = pd.to_numeric(df_2024["Total"], errors="coerce")
unmatched_diff = df_2024.groupby("School")["Total"].mean() - df_2023.groupby("School")["Total"].mean()
print("\nUnmatched Absence Change by School:\n", unmatched_diff)

# Matched group t-test
control_change = matched_df[matched_df["School"] == 0]["ChangeInAbsences"]
treatment_change = matched_df[matched_df["School"] == 1]["ChangeInAbsences"]
t_stat, p_val = ttest_ind(treatment_change, control_change, equal_var=False)
print(f"Matched Change t-test: t={t_stat:.2f}, p={p_val:.4f}")

if p_val < 0.05:
    print("\nInterpretation: There is a statistically significant difference in change of absences between schools after matching. This suggests the policy MAY have had an effect.")
else:
    print("\nInterpretation: There is NO statistically significant difference in change of absences between schools after matching. The policy likely had no measurable impact.")

# =====================
# STEP 4: Visualizations
# =====================

# Interpretation Section
print("\n=== Interpretation of Propensity Score Matching Results ===")
print("Propensity score matching was used to ensure that comparisons between GC and SV students were fair by accounting for differences in demographics and pre-policy attendance.")
print("By matching students with similar characteristics, we isolate the effect of the attendance policy more cleanly.")
print("If the matched comparison shows no significant difference, it suggests that the policy did not have a distinct impact beyond what could be explained by student background.")

# Boxplot side-by-side by school
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.boxplot(ax=axes[0], x="School", y="ChangeInAbsences", data=matched_df)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["GC (Control)", "SV (Policy)"])
axes[0].set_title("Change in Absences (Matched Students)")
axes[0].set_ylabel("Change in Absences")
axes[0].grid(axis="y")

avg_changes = pd.DataFrame({
    "Matched": matched_df.groupby("School")["ChangeInAbsences"].mean(),
    "Unmatched": unmatched_diff
})

avg_changes.T.plot(kind="bar", ax=axes[1])
axes[1].set_title("Average Change in Absences (Matched vs. Unmatched)")
axes[1].set_ylabel("Average Change")
axes[1].grid(axis="y")
axes[1].legend(title="School", labels=["GC (Control)", "SV (Policy)"])

plt.tight_layout()
plt.show()

# Clean bar chart: Average Change in Absences
avg_changes = pd.DataFrame({
    "Matched": matched_df.groupby("School")["ChangeInAbsences"].mean(),
    "Unmatched": unmatched_diff
})

avg_changes.T.plot(kind="bar", figsize=(8, 6))
plt.title("Average Absence Change (Matched vs Unmatched)")
plt.ylabel("Avg. Change in Absences")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.legend(title="School", labels=["GC (Control)", "SV (Policy)"])
plt.tight_layout()
plt.show()

# =====================
# STEP 5: Demographic Breakdown (Race, Gender, ESL)
# =====================

print("\n=== Change in Absences by Demographic Group (Matched Students) ===")

# Combine all race indicators into one categorical column
def reshape_race_data(df):
    race_long = []
    for race in race_cols:
        race_subset = df[df[race] == 1].copy()
        race_subset["Race"] = race
        race_long.append(race_subset)
    if race_long:
        return pd.concat(race_long, ignore_index=True)
    else:
        return pd.DataFrame()

# Plot group differences helper
def plot_group_diffs(df, group_col, title=None, filename="plot.png"):
    groups = df[group_col].dropna().unique()
    plot_data = []
    for val in groups:
        sub = df[df[group_col] == val]
        ctrl = sub[sub["School"] == 0]["ChangeInAbsences"]
        treat = sub[sub["School"] == 1]["ChangeInAbsences"]
        if len(ctrl) >= 3 and len(treat) >= 3:
            plot_data.append({
                group_col: val,
                "GC": ctrl.mean(),
                "SV": treat.mean()
            })
    if plot_data:
        df_plot = pd.DataFrame(plot_data).melt(id_vars=group_col, var_name="School", value_name="AvgChange")
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_plot, x=group_col, y="AvgChange", hue="School")
        plt.title(title or f"Average Change in Absences by {group_col}")
        plt.ylabel("Avg. Change (Post - Pre)")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

# Race plot (combined)
race_data = reshape_race_data(matched_df)
if not race_data.empty and "Race" in race_data.columns:
    race_counts = race_data["Race"].value_counts()
    valid_races = race_counts[race_counts >= 5].index
    race_data = race_data[race_data["Race"].isin(valid_races)]

    race_plot = []
    for r in race_data["Race"].unique():
        group = race_data[race_data["Race"] == r]
        for school in [0, 1]:
            values = group[group["School"] == school]["ChangeInAbsences"]
            if len(values) >= 3:
                race_plot.append({
                    "Race": r,
                    "School": "GC" if school == 0 else "SV",
                    "AvgChange": values.mean()
                })

    race_df = pd.DataFrame(race_plot)
    if not race_df.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=race_df, x="Race", y="AvgChange", hue="School")
        plt.title("Average Change in Absences by Race (Matched Students)")
        plt.ylabel("Avg. Change (Post - Pre)")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig("plot_race.png")
        plt.close()
        print("Saved plot_race.png")

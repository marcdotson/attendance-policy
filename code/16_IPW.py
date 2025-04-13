import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

# =====================
# STEP 1: Load and Preprocess Data (same as propensity score matching)
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
# STEP 2: Calculating Propensity Scores for IPW
# =====================
print("Calculating propensity scores for IPW...")

# Convert and clean features as before
race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]
X = df_2023[[
    "Gender", "ESL", "Grade Level", "Hispanic", "White", "Black",
    "Asian", "Pacific Islander", "American Indian", "Total"
]].copy()
X["Gender"] = X["Gender"].map({"M": 0, "F": 1}).fillna(0)
X["ESL"] = X["ESL"].apply(lambda x: 1 if str(x).upper() == "Y" else 0)
X["Grade Level"] = pd.to_numeric(X["Grade Level"], errors="coerce").fillna(9)
X["Total"] = pd.to_numeric(X["Total"], errors="coerce")
X["Total"] = X["Total"].fillna(X["Total"].median())

for col in race_cols:
    X[col] = X[col].apply(lambda x: 1 if str(x).upper() == "Y" else 0)

# Calculate propensity scores for all students
logit = LogisticRegression()
y = df_2023["School"]
propensity_scores = logit.fit(X, y).predict_proba(X)[:, 1]
df_2023["PropensityScore"] = propensity_scores

# =====================
# STEP 3: Create IPW Weights
# =====================
print("Creating IPW weights...")

# Create weights: Inverse of propensity score for treated, inverse of (1-propensity) for control
df_2023["IPW_Weight"] = np.where(
    df_2023["School"] == 1,
    1 / df_2023["PropensityScore"],
    1 / (1 - df_2023["PropensityScore"])
)

# Trim extreme weights to reduce variance (common practice)
lower_bound = np.percentile(df_2023["IPW_Weight"], 1)
upper_bound = np.percentile(df_2023["IPW_Weight"], 99)
df_2023["IPW_Weight_Trimmed"] = df_2023["IPW_Weight"].clip(lower_bound, upper_bound)

# Normalize weights to sum to sample size
df_2023["IPW_Weight_Normalized"] = df_2023["IPW_Weight_Trimmed"] * len(df_2023) / df_2023["IPW_Weight_Trimmed"].sum()

# =====================
# STEP 4: Prepare DataFrame for Analysis
# =====================
print("Preparing data for IPW analysis...")

# Convert absences to numeric
df_2023["Total"] = pd.to_numeric(df_2023["Total"], errors="coerce")
df_2024["Total"] = pd.to_numeric(df_2024["Total"], errors="coerce")

# Create a merged dataset with pre and post outcomes
analysis_df = df_2023[["Number", "School", "Total", "PropensityScore", "IPW_Weight_Normalized"] + race_cols + ["Gender", "ESL"]]
analysis_df = analysis_df.rename(columns={"Total": "PrePolicyAbsences"})

# Merge with post-policy data
analysis_df = analysis_df.merge(
    df_2024[["Number", "Total"]].rename(columns={"Total": "PostPolicyAbsences"}),
    on="Number",
    how="inner"
)

# Calculate the change in absences
analysis_df["ChangeInAbsences"] = analysis_df["PostPolicyAbsences"] - analysis_df["PrePolicyAbsences"]
analysis_df.dropna(subset=["ChangeInAbsences"], inplace=True)

# =====================
# STEP 5: Weighted Analysis
# =====================
print("Performing weighted analysis...")

# Calculate weighted means by school
weighted_change = {}
for school in [0, 1]:
    school_data = analysis_df[analysis_df["School"] == school]
    weighted_change[school] = np.average(
        school_data["ChangeInAbsences"],
        weights=school_data["IPW_Weight_Normalized"]
    )

print("\nIPW Weighted Average Change in Absences:")
print(f"GC (Control): {weighted_change[0]:.2f}")
print(f"SV (Policy): {weighted_change[1]:.2f}")
print(f"Difference (SV - GC): {weighted_change[1] - weighted_change[0]:.2f}")

# =====================
# STEP 6: Statistical Inference with Weights
# =====================
print("\nPerforming statistical tests with weights...")

# Prepare weighted stats objects
control_data = analysis_df[analysis_df["School"] == 0]
treatment_data = analysis_df[analysis_df["School"] == 1]

control_weighted = DescrStatsW(
    control_data["ChangeInAbsences"], 
    weights=control_data["IPW_Weight_Normalized"],
    ddof=1
)

treatment_weighted = DescrStatsW(
    treatment_data["ChangeInAbsences"],
    weights=treatment_data["IPW_Weight_Normalized"],
    ddof=1
)

# Weighted t-test 
# Use the correct method from statsmodels for weighted t-test
from statsmodels.stats.weightstats import ttest_ind
t_stat, p_val, _ = ttest_ind(
    treatment_data["ChangeInAbsences"], 
    control_data["ChangeInAbsences"],
    weights=(treatment_data["IPW_Weight_Normalized"], control_data["IPW_Weight_Normalized"]),
    usevar='unequal'
)
print(f"Weighted t-test: t={t_stat:.2f}, p={p_val:.4f}")

if p_val < 0.05:
    print("\nInterpretation: There is a statistically significant difference in change of absences between schools using IPW. This suggests the policy MAY have had an effect.")
else:
    print("\nInterpretation: There is NO statistically significant difference in change of absences between schools using IPW. The policy likely had no measurable impact.")

# =====================
# STEP 7: Visualization of IPW Results
# =====================
print("\nCreating visualizations...")

# Create a bar plot of the weighted means
schools = ["GC (Control)", "SV (Policy)"]
plt.figure(figsize=(8, 6))
colors = ['blue', 'orange']
plt.bar(schools, [weighted_change[0], weighted_change[1]], color = colors)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.ylabel("Weighted Average Change in Absences")
plt.title("IPW Analysis: Average Change in Absences by School")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("ipw_main_results.png")
plt.close()

# Distribution of propensity scores by treatment group
plt.figure(figsize=(10, 6))
sns.histplot(data=df_2023, x="PropensityScore", hue="School", bins=30, 
             element="step", stat="density", common_norm=False)
plt.title("Distribution of Propensity Scores by School")
plt.xlabel("Propensity Score")
plt.ylabel("Density")
plt.legend(title="School", labels=["GC (Control)", "SV (Policy)"])
plt.tight_layout()
plt.savefig("ipw_propensity_distribution.png")
plt.close()

# Distribution of weights
plt.figure(figsize=(10, 6))
sns.histplot(data=df_2023, x="IPW_Weight_Normalized", hue="School", bins=30, 
             element="step", stat="density", common_norm=False)
plt.title("Distribution of IPW Weights by School")
plt.xlabel("Normalized IPW Weight")
plt.ylabel("Density")
plt.legend(title="School", labels=["GC (Control)", "SV (Policy)"])
plt.tight_layout()
plt.savefig("ipw_weights_distribution.png")
plt.close()

# =====================
# STEP 8: Demographic Breakdown Analysis with IPW
# =====================
print("\n=== Change in Absences by Demographic Group (IPW Weighted) ===")

# Reshape race data function (similar to your original)
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

# Plot group differences with weights helper
def plot_weighted_group_diffs(df, group_col, title=None, filename="plot.png"):
    groups = df[group_col].dropna().unique()
    plot_data = []
    for val in groups:
        sub = df[df[group_col] == val]
        if len(sub) >= 10:  # Ensure enough data points
            for school in [0, 1]:
                school_data = sub[sub["School"] == school]
                if len(school_data) >= 5:  # Minimum sample
                    weighted_mean = np.average(
                        school_data["ChangeInAbsences"],
                        weights=school_data["IPW_Weight_Normalized"]
                    )
                    plot_data.append({
                        group_col: val,
                        "School": "GC" if school == 0 else "SV",
                        "WeightedAvgChange": weighted_mean
                    })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_plot, x=group_col, y="WeightedAvgChange", hue="School")
        plt.title(title or f"Weighted Average Change in Absences by {group_col}")
        plt.ylabel("Weighted Avg. Change (Post - Pre)")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

# Race plot (combined)
race_data = reshape_race_data(analysis_df)
if not race_data.empty and "Race" in race_data.columns:
    race_counts = race_data["Race"].value_counts()
    valid_races = race_counts[race_counts >= 10].index
    race_data = race_data[race_data["Race"].isin(valid_races)]
    
    plot_weighted_group_diffs(race_data, "Race", 
                              title="IPW: Weighted Change in Absences by Race", 
                              filename="ipw_race.png")

# Gender plot
plot_weighted_group_diffs(analysis_df, "Gender", 
                         title="IPW: Weighted Change in Absences by Gender", 
                         filename="ipw_gender.png")

# ESL plot
plot_weighted_group_diffs(analysis_df, "ESL", 
                         title="IPW: Weighted Change in Absences by ESL Status", 
                         filename="ipw_esl.png")

# Print summary for reporting
print("\n=== SUMMARY OF IPW ANALYSIS ===")
print(f"Total students analyzed: {len(analysis_df)}")
print(f"GC (Control) students: {len(analysis_df[analysis_df['School'] == 0])}")
print(f"SV (Policy) students: {len(analysis_df[analysis_df['School'] == 1])}")
print(f"\nPre-policy average absences:")
print(f"GC: {analysis_df[analysis_df['School'] == 0]['PrePolicyAbsences'].mean():.2f}")
print(f"SV: {analysis_df[analysis_df['School'] == 1]['PrePolicyAbsences'].mean():.2f}")
print(f"\nPost-policy average absences:")
print(f"GC: {analysis_df[analysis_df['School'] == 0]['PostPolicyAbsences'].mean():.2f}")
print(f"SV: {analysis_df[analysis_df['School'] == 1]['PostPolicyAbsences'].mean():.2f}")
print(f"\nWeighted change in absences:")
print(f"GC: {weighted_change[0]:.2f}")
print(f"SV: {weighted_change[1]:.2f}")
print(f"Difference: {weighted_change[1] - weighted_change[0]:.2f}")
print(f"Statistical significance: {'Yes' if p_val < 0.05 else 'No'} (p={p_val:.4f})")

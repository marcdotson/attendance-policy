#----------------------------------------
# Purpose: Propensity Score Matching Baseline (vs Naive Baseline)
#----------------------------------------
#
# This script uses *propensity score matching* (PSM) to match similar students 
# between GC (control) and SV (treatment) based on demographics (gender, race, ESL, grade) and pre-policy absences.
# 
# In contrast to the naive baseline (which simply averages absences across schools),
# this matching approach adjusts for selection bias and provides stronger causal evidence
# about the effect of the attendance policy.
#
# This script handles Trimester 1 (T1), Trimester 2 (T2), and is expandable for future T3.
#
#----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind

#----------------------------------------
# Load Data
#----------------------------------------

# Load demographics
demographics = pd.read_csv('data/Attendance Demographics 2024-2025.csv', dtype=str, low_memory=False)
demographics.rename(columns={'Student Number': 'Number'}, inplace=True)

# Define trimester files
attendance_files = {
    'T1': [
        ('data/23-24 T1 Attendance.xlsx', 'GC - Absence', 0, 2023),
        ('data/23-24 T1 Attendance.xlsx', 'SV - Absence', 1, 2023),
        ('data/24-25 T1 Attendance.xlsx', 'GC - Absences', 0, 2024),
        ('data/24-25 T1 Attendance.xlsx', 'SV - Absences', 1, 2024)
    ],
    'T2': [
        ('data/23-24 T2 Attendance.xlsx', 'GC - Absence', 0, 2023),
        ('data/23-24 T2 Attendance.xlsx', 'SV - Absence', 1, 2023),
        ('data/24-25 T2 Attendance.xlsx', 'GC - Absences', 0, 2024),
        ('data/24-25 T2 Attendance.xlsx', 'SV - Absences', 1, 2024)
    ]
    # Add T3 here when available
}

#----------------------------------------
# Matching Function
#----------------------------------------

def run_matching(trimester_label, files):
    print(f"\n=== Running Matching for {trimester_label} ===")

    dfs_2023, dfs_2024 = [], []
    for file, sheet, school, year in files:
        df = pd.read_excel(file, sheet_name=sheet, dtype=str)
        df.columns = df.columns.map(str).str.strip()
        df['Number'] = df['Number'].astype(str)
        df['School_from_attendance'] = school
        df['Year'] = year
        df['Trimester'] = trimester_label
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
        df = df.merge(demographics, on='Number', how='left')
        df = df.dropna(subset=['Total', 'Grade Level'])  # Drop rows with missing Total or Grade Level
        if year == 2023:
            dfs_2023.append(df)
        else:
            dfs_2024.append(df)

    df_2023 = pd.concat(dfs_2023, ignore_index=True)
    df_2024 = pd.concat(dfs_2024, ignore_index=True)

    # Preprocessing matching variables
    df_2023['Gender'] = df_2023['Gender'].map({'M': 0, 'F': 1})
    df_2023['ESL'] = df_2023['ESL'].map({'Y': 1, 'N': 0})
    df_2023['SWD'] = df_2023['SWD'].map({'Y': 1, 'N': 0})
    df_2023['PrePolicyAbsences'] = df_2023['Total']

    race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]

    X = df_2023[['Gender', 'ESL', 'Grade Level'] + race_cols + ['PrePolicyAbsences']].copy()
    X['Gender'] = pd.to_numeric(X['Gender'], errors='coerce').fillna(0).astype(int)
    X['ESL'] = pd.to_numeric(X['ESL'], errors='coerce').fillna(0).astype(int)
    X['Grade Level'] = pd.to_numeric(X['Grade Level'], errors='coerce').fillna(11).astype(int)
    for col in race_cols:
        X[col] = X[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
    X['PrePolicyAbsences'] = pd.to_numeric(X['PrePolicyAbsences'], errors='coerce').fillna(X['PrePolicyAbsences'].median())

    # Fit logistic regression
    logit = LogisticRegression()
    df_2023['PropensityScore'] = logit.fit(X, df_2023['School_from_attendance']).predict_proba(X)[:, 1]

    # Nearest neighbor matching
    treated = df_2023[df_2023['School_from_attendance'] == 1]
    control = df_2023[df_2023['School_from_attendance'] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['PropensityScore']])
    distances, indices = nn.kneighbors(treated[['PropensityScore']])

    matched_control = control.iloc[indices.flatten()]
    matched_data = pd.concat([treated, matched_control])

    # 2024 Matched Data
    df_2024_matched = df_2024[df_2024['Number'].isin(matched_data['Number'])].copy()
    df_2024_matched['Total'] = pd.to_numeric(df_2024_matched['Total'], errors='coerce')
    df_2024_matched.dropna(subset=['Total'], inplace=True)

    return df_2024_matched

#----------------------------------------
# Run Matching for Each Trimester
#----------------------------------------

all_matched = []
for trimester_label, files in attendance_files.items():
    matched = run_matching(trimester_label, files)
    all_matched.append(matched)

df_matched_combined = pd.concat(all_matched, ignore_index=True)

#----------------------------------------
# Analyze Matched Results
#----------------------------------------

colors = ['green', 'blue']

# Barplot: Group Means
plt.figure(figsize=(8, 6))
sns.barplot(x='Trimester', y='Total', hue='School_from_attendance', data=df_matched_combined, palette=colors, estimator=np.mean, errorbar=None)
plt.title("Average Post-Policy Absences by Trimester (Matched Students)")
plt.ylabel("Average Absences")
plt.xlabel("Trimester")
plt.legend(title="School", labels=["GC", "SV"])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#----------------------------------------
# Break Down by Grade
#----------------------------------------

df_matched_combined['Grade Level'] = pd.to_numeric(df_matched_combined['Grade Level'], errors='coerce')

grade_grouped = df_matched_combined[df_matched_combined['Grade Level'].isin([10, 11, 12])]
grade_grouped = grade_grouped.groupby(['Grade Level', 'School_from_attendance'])['Total'].mean().unstack().fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
grade_grouped.plot(kind='bar', ax=ax, color=colors)
plt.title("Average Absences by Grade Level (Matched Students)")
plt.ylabel("Average Absences")
plt.xlabel("Grade Level")
plt.xticks(rotation=0, ticks=[0,1,2], labels=["10th", "11th", "12th"])
plt.legend(["GC", "SV"], title="School")
plt.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

#----------------------------------------
# Break Down by Race
#----------------------------------------

race_cols = ["Hispanic", "White", "Black", "Asian", "Pacific Islander", "American Indian"]
for col in race_cols:
    df_matched_combined[col] = df_matched_combined[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

race_avgs = []
for race in race_cols:
    gc_vals = df_matched_combined[(df_matched_combined[race] == 1) & (df_matched_combined['School_from_attendance'] == 0)]['Total']
    sv_vals = df_matched_combined[(df_matched_combined[race] == 1) & (df_matched_combined['School_from_attendance'] == 1)]['Total']

    race_avgs.append({
        'Race': race,
        'GC': gc_vals.mean(),
        'SV': sv_vals.mean()
    })

race_df = pd.DataFrame(race_avgs).melt(id_vars='Race', var_name='School', value_name='Average Absences')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Race', y='Average Absences', hue='School', data=race_df, palette=colors)
plt.title("Average Absences by Race (Matched Students)")
plt.xticks(rotation=45)
plt.ylabel("Average Absences")
plt.xlabel("Race")
plt.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', 
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

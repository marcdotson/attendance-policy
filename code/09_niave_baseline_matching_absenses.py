#----------------------------------------
# Naive Baseline Matching: Multi-Trimester Auto Analysis (T1, T2, T3 Ready)
#----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind

#----------------------------------------
# 1. Load Attendance Data (T1 + T2, ready for T3)
#----------------------------------------

def load_attendance_data():
    # T1
    gc_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence', dtype=str)
    sv_2023_t1 = pd.read_excel('data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence', dtype=str)
    gc_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences', dtype=str)
    sv_2024_t1 = pd.read_excel('data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences', dtype=str)

    # T2
    gc_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='GC - Absence', dtype=str)
    sv_2023_t2 = pd.read_excel('data/23-24 T2 Attendance.xlsx', sheet_name='SV - Absence', dtype=str)
    gc_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='GC - Absences', dtype=str)
    sv_2024_t2 = pd.read_excel('data/24-25 T2 Attendance.xlsx', sheet_name='SV - Absences', dtype=str)

    # Add school and trimester tags
    for df, school in zip(
        [gc_2023_t1, sv_2023_t1, gc_2024_t1, sv_2024_t1, gc_2023_t2, sv_2023_t2, gc_2024_t2, sv_2024_t2],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ):
        df['School_from_attendance'] = school

    for df in [gc_2023_t1, sv_2023_t1, gc_2024_t1, sv_2024_t1]:
        df['Trimester'] = 'T1'
    for df in [gc_2023_t2, sv_2023_t2, gc_2024_t2, sv_2024_t2]:
        df['Trimester'] = 'T2'

    # Clean column names
    for df in [gc_2023_t1, sv_2023_t1, gc_2024_t1, sv_2024_t1, gc_2023_t2, sv_2023_t2, gc_2024_t2, sv_2024_t2]:
        df.columns = df.columns.map(str).str.strip()
    
    df_2023 = pd.concat([gc_2023_t1, sv_2023_t1, gc_2023_t2, sv_2023_t2], ignore_index=True)
    df_2024 = pd.concat([gc_2024_t1, sv_2024_t1, gc_2024_t2, sv_2024_t2], ignore_index=True)

    return df_2023, df_2024

#----------------------------------------
# 2. Load Demographics
#----------------------------------------

def load_demographics():
    demographics = pd.read_csv('data/Attendance Demographics 2024-2025.csv')
    demographics['Student Number'] = demographics['Student Number'].astype(str)
    return demographics

#----------------------------------------
# 3. Merge Data
#----------------------------------------

def merge_data(attendance, demographics):
    attendance['Number'] = attendance['Number'].astype(str)
    merged = attendance.merge(demographics, left_on='Number', right_on='Student Number', how='left')
    return merged

#----------------------------------------
# 4. Matching and Analysis for One Trimester
#----------------------------------------

def run_matching_analysis(df_2023, df_2024, trimester='T2'):
    print(f"\nRunning analysis for {trimester}...")

    df_2023_trim = df_2023[df_2023['Trimester'] == trimester].copy()
    df_2024_trim = df_2024[df_2024['Trimester'] == trimester].copy()

    # Preprocess matching variables
    df_2023_trim['Gender'] = df_2023_trim['Gender'].map({'M': 0, 'F': 1})
    df_2023_trim['ESL'] = df_2023_trim['ESL'].map({'Y': 1, 'N': 0})
    df_2023_trim['SWD'] = df_2023_trim['SWD'].map({'Y': 1, 'N': 0})
    df_2023_trim['PrePolicyAbsences'] = pd.to_numeric(df_2023_trim['Total'], errors='coerce')

    X = df_2023_trim[[ 'Gender', 'ESL', 'Grade Level', 'Hispanic', 'White', 'Black', 'Asian', 'Pacific Islander', 'American Indian', 'PrePolicyAbsences']].copy()

    X['Gender'] = X['Gender'].fillna(0).astype(int)
    X['ESL'] = X['ESL'].fillna(0).astype(int)
    X['Grade Level'] = pd.to_numeric(X['Grade Level'], errors='coerce').fillna(X['Grade Level'].median())

    for col in ['Hispanic', 'White', 'Black', 'Asian', 'Pacific Islander', 'American Indian']:
        X[col] = X[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    X['PrePolicyAbsences'] = X['PrePolicyAbsences'].fillna(X['PrePolicyAbsences'].median())

    # Fit propensity score model
    logit = LogisticRegression()
    df_2023_trim['PropensityScore'] = logit.fit(X, df_2023_trim['School_from_attendance']).predict_proba(X)[:, 1]

    # Nearest neighbor matching
    treated = df_2023_trim[df_2023_trim['School_from_attendance'] == 1]
    control = df_2023_trim[df_2023_trim['School_from_attendance'] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['PropensityScore']])
    distances, indices = nn.kneighbors(treated[['PropensityScore']])

    matched_control = control.iloc[indices.flatten()]
    matched_data = pd.concat([treated, matched_control])

    # Post-policy matched
    df_2024_matched = df_2024_trim[df_2024_trim['Number'].isin(matched_data['Number'])].copy()
    df_2024_matched['Total'] = pd.to_numeric(df_2024_matched['Total'], errors='coerce')
    df_2024_matched = df_2024_matched.dropna(subset=['Total'])
    df_2024_matched['Trimester'] = trimester

    # T-test
    t_stat, p_value = ttest_ind(
        df_2024_matched[df_2024_matched['School_from_attendance'] == 1]['Total'],
        df_2024_matched[df_2024_matched['School_from_attendance'] == 0]['Total'],
        equal_var=False
    )

    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    return df_2024_matched

#----------------------------------------
# 5. Plot Combined Results (T1 + T2)
#----------------------------------------

def plot_combined(df_combined):
    colors = ['green', 'blue']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Trimester", y="Total", hue="School_from_attendance", data=df_combined, palette=colors, estimator=np.mean, ci=None)
    ax.set_title("Average Post-Policy Absences by Trimester (GC vs SV)")
    ax.set_ylabel("Average Absences")
    ax.set_xlabel("Trimester")
    plt.legend(title="School", labels=["GC", "SV"])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

#----------------------------------------
# 6. Plot Gender Distribution (NEW)
#----------------------------------------

def plot_gender_distribution(df_combined):
    colors = ['green', 'blue']

    # Only keep valid genders
    gender_valid = df_combined[df_combined['Gender'].isin(['M', 'F'])].copy()

    # Group by Gender and School
    gender_grouped = gender_valid.groupby(['Gender', 'School_from_attendance'])['Total'].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 6))
    gender_grouped.plot(kind='bar', ax=ax, color=colors)
    plt.title("Average Absences by Gender (GC vs SV)")
    plt.ylabel("Average Absences")
    plt.xlabel("Gender")
    plt.xticks(rotation=0)

    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    plt.legend(["GC", "SV"], title="School")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

#----------------------------------------
# 7. MAIN EXECUTION
#----------------------------------------

df_2023, df_2024 = load_attendance_data()
demographics = load_demographics()

df_2023 = merge_data(df_2023, demographics)
df_2024 = merge_data(df_2024, demographics)

all_trimesters = ['T1', 'T2']  # Add 'T3' when ready

matched_list = []
for tri in all_trimesters:
    matched_df = run_matching_analysis(df_2023, df_2024, trimester=tri)
    matched_list.append(matched_df)

df_combined_matched = pd.concat(matched_list, ignore_index=True)

# Final plots
plot_combined(df_combined_matched)
plot_gender_distribution(df_combined_matched)

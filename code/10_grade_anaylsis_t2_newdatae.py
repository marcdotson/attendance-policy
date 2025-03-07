import pandas as pd

# Load attendance data for both trimesters
gc_2023_t1 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t1 = pd.read_excel('./data/23-24 T1 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t1 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t1 = pd.read_excel('./data/24-25 T1 Attendance.xlsx', sheet_name='SV - Absences')

gc_2023_t2 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='GC - Absence')
sv_2023_t2 = pd.read_excel('./data/23-24 T2 Attendance.xlsx', sheet_name='SV - Absence')
gc_2024_t2 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='GC - Absences')
sv_2024_t2 = pd.read_excel('./data/24-25 T2 Attendance.xlsx', sheet_name='SV - Absences')

# Load new dataset with current grades
current_grades = pd.read_csv("./data/Attendance Demographics 2024-2025.csv", dtype=str, low_memory=False)  # <- Replace with actual file path
current_grades.rename(columns={'Student Number': 'Number'}, inplace=True)  # Rename to match attendance data

# Merge attendance data with current grades
for df_name, df in zip(["gc_2023_t1", "sv_2023_t1", "gc_2024_t1", "sv_2024_t1",
                         "gc_2023_t2", "sv_2023_t2", "gc_2024_t2", "sv_2024_t2"],
                         [gc_2023_t1, sv_2023_t1, gc_2024_t1, sv_2024_t1,
                          gc_2023_t2, sv_2023_t2, gc_2024_t2, sv_2024_t2]):
    if 'number' in df.columns:  # Rename if lowercase
        df.rename(columns={'number': 'Number'}, inplace=True)
    df['Number'] = df['Number'].astype(str)  # Ensure consistency
    
    # Merge with current grade data
    df = df.merge(current_grades[['Number', 'Grade Level']], on='Number', how='left')

    # Convert 'Grade Level' to numeric
    df['Grade Level'] = pd.to_numeric(df['Grade Level'], errors='coerce')

    # Store back to the variable
    locals()[df_name] = df

# Process all data for Trimester 1 first
print("=== Trimester 1 ===")
for trimester in ["t1"]:  # Only process Trimester 1
    for grade in range(9, 13):  # Loop through grade levels (9-12)
        # Calculate average total absences for each group
        gc_23_avg = locals()[f"gc_2023_{trimester}"][locals()[f"gc_2023_{trimester}"]['Grade Level'] == grade]['Total'].mean()
        gc_24_avg = locals()[f"gc_2024_{trimester}"][locals()[f"gc_2024_{trimester}"]['Grade Level'] == grade]['Total'].mean()
        sv_23_avg = locals()[f"sv_2023_{trimester}"][locals()[f"sv_2023_{trimester}"]['Grade Level'] == grade]['Total'].mean()
        sv_24_avg = locals()[f"sv_2024_{trimester}"][locals()[f"sv_2024_{trimester}"]['Grade Level'] == grade]['Total'].mean()

        # Difference-in-Difference (DID) for total absences
        gc_diff = gc_24_avg - gc_23_avg
        sv_diff = sv_24_avg - sv_23_avg
        did = gc_diff - sv_diff

        print(f'Grade {grade}: DID in mean total absences is {did:.2f}')

        # Loop through periods (1-5)
        for period in range(1, 6):
            gc_23_avg = locals()[f"gc_2023_{trimester}"][locals()[f"gc_2023_{trimester}"]['Grade Level'] == grade][period].mean()
            gc_24_avg = locals()[f"gc_2024_{trimester}"][locals()[f"gc_2024_{trimester}"]['Grade Level'] == grade][period].mean()
            sv_23_avg = locals()[f"sv_2023_{trimester}"][locals()[f"sv_2023_{trimester}"]['Grade Level'] == grade][period].mean()
            sv_24_avg = locals()[f"sv_2024_{trimester}"][locals()[f"sv_2024_{trimester}"]['Grade Level'] == grade][period].mean()

            # Difference-in-Difference (DID) for absences by period
            gc_diff = gc_24_avg - gc_23_avg
            sv_diff = sv_24_avg - sv_23_avg
            did = gc_diff - sv_diff

            print(f'Grade {grade}: DID in mean absences (period {period}) is {did:.2f}')

# Now process all data for Trimester 2

print("\n\n=== Trimester 2 ===")
for trimester in ["t2"]:  # Only process Trimester 2
    for grade in range(9, 13):  # Loop through grade levels (9-12)
        # Calculate average total absences for each group
        gc_23_avg = locals()[f"gc_2023_{trimester}"][locals()[f"gc_2023_{trimester}"]['Grade Level'] == grade]['Total'].mean()
        gc_24_avg = locals()[f"gc_2024_{trimester}"][locals()[f"gc_2024_{trimester}"]['Grade Level'] == grade]['Total'].mean()
        sv_23_avg = locals()[f"sv_2023_{trimester}"][locals()[f"sv_2023_{trimester}"]['Grade Level'] == grade]['Total'].mean()
        sv_24_avg = locals()[f"sv_2024_{trimester}"][locals()[f"sv_2024_{trimester}"]['Grade Level'] == grade]['Total'].mean()

        # Difference-in-Difference (DID) for total absences
        gc_diff = gc_24_avg - gc_23_avg
        sv_diff = sv_24_avg - sv_23_avg
        did = gc_diff - sv_diff

        print(f'Grade {grade}: DID in mean total absences is {did:.2f}')

        # Loop through periods (1-5)
        for period in range(1, 6):
            gc_23_avg = locals()[f"gc_2023_{trimester}"][locals()[f"gc_2023_{trimester}"]['Grade Level'] == grade][period].mean()
            gc_24_avg = locals()[f"gc_2024_{trimester}"][locals()[f"gc_2024_{trimester}"]['Grade Level'] == grade][period].mean()
            sv_23_avg = locals()[f"sv_2023_{trimester}"][locals()[f"sv_2023_{trimester}"]['Grade Level'] == grade][period].mean()
            sv_24_avg = locals()[f"sv_2024_{trimester}"][locals()[f"sv_2024_{trimester}"]['Grade Level'] == grade][period].mean()

            # Difference-in-Difference (DID) for absences by period
            gc_diff = gc_24_avg - gc_23_avg
            sv_diff = sv_24_avg - sv_23_avg
            did = gc_diff - sv_diff

            print(f'Grade {grade}: DID in mean absences (period {period}) is {did:.2f}')

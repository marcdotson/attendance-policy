# Attendance Policy Evaluation


## Description

This project evaluates the effectiveness of High School A’s attendance
policy implemented during the first trimester of the 2024-2025 school
year. Not only can we compare the change in attendance from the first
trimester to the previous year’s attendance at High School A, the
implementation of the policy by High School A alone serves as a kind of
natural experiment, allowing comparison to High School B’s attendance
during the same trimester where High School B serves as a control group.
We will evaluate how the attendance policy has impacted different groups
of students (e.g., demographic/income splits) by these same comparisons
in and across high schools.

## Project Organization

- `/code` Scripts with prefixes (e.g., `01_import-data.py`,
  `02_clean-data.py`) and functions in `/code/src`.
- `/data` Simulated and real data, the latter not pushed.
- `/figures` PNG images and plots.
- `/output` Output from model runs, not pushed.
- `/presentations` Presentation slides.
- `/private` A catch-all folder for miscellaneous files, not pushed.
- `/writing` Reports, posts, and case studies.
- `/.venv` Hidden project library, not pushed.
- `.gitignore` Hidden Git instructions file.
- `.python-version` Hidden Python version for the reproducible
  environment.
- `requirements.txt` Information on the reproducible environment.

## Reproducible Environment

After cloning this repository, go to the project’s terminal in Positron
and run `python -m venv .venv` to create the `/.venv` project library,
followed by `pip install -r requirements.txt` to install the specified
library versions.

Whenever you install new libraries or decide to update the versions of
libraries you use, run `pip freeze > requirements.txt` to update
`requirements.txt`.

For more details on using GitHub, Quarto, etc. see [ASC
Training](https://github.com/marcdotson/asc-training).

# Diff-In-Diff-Notes


The difference-in-differences method is a quasi-experimental approach
that compares the changes in outcomes over time between a population
enrolled in a program (the treatment group) and a population that is not
(the comparison group). It is a useful tool for data analysis.

The dataset is adapted from the dataset in Card and Krueger (1994),
which estimates the causal effect of an increase in the state minimum
wage on the employment.

- On April 1, 1992, New Jersey raised the state minimum wage from 4.25
  USD to 5.05 USD while the minimum wage in Pennsylvania stays the same
  at 4.25 USD.
- Data about employment in fast-food restaurants in NJ (0) and PA (1)
  were collected in February 1992 and in November 1992.
- 384 restaurants in total after removing null values

The calculation of DID is simple:

- mean PA (control group) employee per restaurant before/after the
  treatment is 23.38/21.1, so the after/before difference for the
  control group is -2.28 (21.1 - 23.38)
- mean NJ (treatment group) employee per restaurant before/after the
  treatment is 20.43/20.90, so the after/before difference for the
  treatment group is 0.47 (20.9 - 20.43)
- the difference-in-differences (DID) is 2.75 (0.47 + 2.28), which is
  (the after/before difference of the treatment group) - (the
  after/before difference of the control group)

``` python
import pandas as pd
df = pd.read_csv('FILE PATH')

df.info()
df.head()
df.groupby('state').mean()

# check by calculating the mean for each group directly
# 0 PA control group, 1 NJ treatment group
mean_emp_pa_before = df.groupby('state').mean().iloc[0, 0]
mean_emp_pa_after = df.groupby('state').mean().iloc[0, 1]
mean_emp_nj_before = df.groupby('state').mean().iloc[1, 0]
mean_emp_nj_after = df.groupby('state').mean().iloc[1, 1]

print(f'mean PA employment before: {mean_emp_pa_before:.2f}')
print(f'mean PA employment after: {mean_emp_pa_after:.2f}')
print(f'mean NJ employment before: {mean_emp_nj_before:.2f}')
print(f'mean NJ employment after: {mean_emp_nj_after:.2f}')

pa_diff = mean_emp_pa_after - mean_emp_pa_before
nj_diff = mean_emp_nj_after - mean_emp_nj_before
did = nj_diff - pa_diff

print(f'DID in mean employment is {did:.2f}')

# group g: 0 control group (PA), 1 treatment group (NJ)
# t: 0 before treatment (min wage raise), 1 after treatment
# gt: interaction of g * t

# data before the treatment
df_before = df[['total_emp_feb', 'state']]
df_before['t'] = 0
df_before.columns = ['total_emp', 'g', 't']

# data after the treatment
df_after = df[['total_emp_nov', 'state']]
df_after['t'] = 1
df_after.columns = ['total_emp', 'g', 't']

# data for regression
df_reg = pd.concat([df_before, df_after])

# create the interaction
df_reg['gt'] = df_reg.g * df_reg.t

df_reg

# regression via sklearn
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

X = df_reg[['g', 't', 'gt']]
y = df_reg.total_emp

lr.fit(X, y)
lr.coef_  # the coefficient for gt is the DID, which is 2.75
# regression via statsmodels
# result is not significant

from statsmodels.formula.api import ols
ols = ols('total_emp ~ g + t + gt', data=df_reg).fit()
print(ols.summary())
```

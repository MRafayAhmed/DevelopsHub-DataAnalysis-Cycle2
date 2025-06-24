import matplotlib.style
import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import re 
from scipy import stats
from functools import reduce 
 
matplotlib 
matplotlib.style.use("ggplot") 
pd.set_option('display.max_colwidth', None )
pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000
sns.set_style("whitegrid")

hr = pd.read_csv("C:/Users/Administrator/.cache/kagglehub/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/versions/1/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(hr.shape)
before_dedup = hr.shape[0]
hr.describe(include= 'all')

#Data Cleaning 
#Check missing value 
print(np.count_nonzero(hr.isnull().values))
print(hr.isnull().any())

#check for duplicates
print(hr[hr.duplicated(keep = False)].shape)

# Strip whitespaces
hr = hr.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Check for conflicting types
hr.dtypes

# Hate to throw away data but it's only 353 values out of over 23 thousand
hr.dropna(axis=0, inplace=True)

# Get rid of all the duplicates
hr.drop_duplicates(inplace=True)

# Lets see what it looks like now
print("Duplicates Removed: " + str(before_dedup - hr.shape[0]))
hr.describe()

hr['JobSatisfaction'].value_counts()

hr['JobSatisfaction'].unique()
# Half the rows in JobSatisfaction seem to be strings. 
# It's the same for the other columns. Let's cast them to floats.
cols = ['JobSatisfaction', 'HourlyRate', 'MonthlyIncome', 'PercentSalaryHike']
hr[cols] = hr[cols].astype(np.float64)

# I know from looking in Excel that certain fields are useless so lets get rid of them
hr = hr.drop(['EmployeeCount', 'Over18', "StandardHours", "EmployeeNumber"], axis= 1)
for col in hr:
    print(col)
    print(hr[col].unique()) 
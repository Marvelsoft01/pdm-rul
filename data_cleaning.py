import pandas as pd
file_path = "dataset/PM_train.txt"

df_train = pd.read_csv(file_path, delimiter=" " , header =None)
#delimiter : this tells python how the columns are separated in the file
""" 
  (a single space) means columns are separated by spaces. thats why we used delimiter=" "

  If the file was comma-separated, we would use delimiter=",".

 If it was tab-separated, we would use delimiter="\t".
"""
#header=none : means the file has no column names in the first row to remove first row column names

df_train  =  df_train.dropna(axis=1, how="all")
#dropna method  : this functions removes missing values(NAN) from the DataFrame


#axis parameter : Tells pandas what to remove
"""
axis=0 → Drops rows with NaN values.

axis=1 → Drops columns with NaN values. 
"""


#how parameter: Tells pandas when to drop

"""
all → Drop a column only if every value in that column is NaN. 

any" → Drop a column if at least one NaN is present.
"""

print(df_train.head())
#head: shows the firt 5 rows, to know what the Dataframe looks like after loading

print(df_train.isna().sum())
#see missing values in each column

print(df_train.shape)
#shape: checks the number of rows and columns

print(df_train.info())  
#info: Checks for missing values and data types
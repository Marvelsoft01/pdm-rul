"""
compute_rul.py

This script defines a reusable function to compute the Remaining Useful Life (RUL)
for each row in a dataset of engine or machine cycles.

The function:
- Groups the dataset by unit ID to find the maximum cycle for each engine.
- Calculates RUL as the difference between the max cycle and the current cycle.
- Appends a new RUL column to the original DataFrame.

This function is used to prepare the target variable for model training and evaluation.
"""

import pandas as pd

def compute_rul(df):
    """Compute Remaining Useful Life (RUL) for a dataset."""
    rul_df = df.groupby('unit')['cycle'].max().reset_index()
    rul_df.columns = ['unit', 'max_cycle']
    
    df = df.merge(rul_df, on='unit', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    
    return df

# Example Usage
# df_train = compute_rul(df_train)
# df_test = compute_rul(df_test)

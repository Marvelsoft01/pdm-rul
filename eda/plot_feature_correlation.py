"""
What This Script Does:
Shows correlation of ALL sensors with RUL, highlighting the Top 10 in bold purple.
Used to justify feature reduction process (all → top 10 → top 5).
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Access config from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from config import TRAIN_WITH_RUL

# === Load Data
df = pd.read_csv(TRAIN_WITH_RUL)

# === Correlation with RUL
corr = df.drop(columns=["unit", "cycle"]).corr()
rul_corr = corr['RUL'].drop('RUL').sort_values(key=abs, ascending=False)

# === Identify top 10 sensor features
top_10_features = rul_corr.head(10).index.tolist()

# === Set color palette (highlight top 10)
colors = ['#8e44ad' if feature in top_10_features else '#dcdcdc' for feature in rul_corr.index]

# === Plot
plt.figure(figsize=(12, 8))
sns.barplot(
    x=rul_corr.values,
    y=rul_corr.index,
    palette=colors
)

plt.title("All Sensor Correlations with RUL\n(Top 10 Highlighted in Purple)", fontsize=14)
plt.xlabel("Correlation Coefficient (Absolute)")
plt.ylabel("Sensor")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

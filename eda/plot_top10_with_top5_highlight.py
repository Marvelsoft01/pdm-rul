"""
What This Script Does:
Plots the top 10 sensor features most correlated with RUL,
while visually highlighting the top 5 that were actually used in the model.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔧 Allow imports from parent folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# ✅ Now import config path safely
from config import TRAIN_WITH_RUL

# === Load Data
df = pd.read_csv(TRAIN_WITH_RUL)

# === Compute Correlation with RUL
corr_matrix = df.drop(columns=['unit', 'cycle']).corr()
rul_corr = corr_matrix['RUL'].drop('RUL').sort_values(key=abs, ascending=False)

# === Extract Top 10 Features
top_10_features = rul_corr.head(10)

# === Define Top 5 Actually Used in Model
top_5_used = ['sensor_14', 'sensor_7', 'sensor_18', 'sensor_24', 'sensor_6']

# === Assign Colors: Highlight top 5
colors = ['#8e44ad' if feature in top_5_used else 'gray' for feature in top_10_features.index]

# === Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_10_features.values,
    y=top_10_features.index,
    palette=colors
)

plt.title("Top 10 Sensors by Correlation with RUL\n(Purple = Used in Model)", fontsize=14)
plt.xlabel("Correlation with RUL")
plt.ylabel("Sensor Feature")
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

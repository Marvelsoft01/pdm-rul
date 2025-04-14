"""
 What This Script Does:
Plots the Remaining Useful Life (RUL) trend for a selected unit over time (cycle).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ✅ Add root folder to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from config import TRAIN_WITH_RUL  # Now this will work

# === Load Training Data with RUL
df_train = pd.read_csv(TRAIN_WITH_RUL)

# === Check if RUL exists
if "RUL" not in df_train.columns:
    raise ValueError("❌ RUL column is missing. Ensure compute_rul.py has been run.")

# === Pick one unit to show the RUL trend
unit_id = 1
unit_data = df_train[df_train["unit"] == unit_id]

# === Plot
plt.figure(figsize=(10, 5))
sns.set(style="darkgrid")

plt.plot(unit_data["cycle"], unit_data["RUL"], marker="o", linestyle="-", color="#8e44ad", linewidth=2)

plt.title(f"RUL Trend for Unit {unit_id}", fontsize=14, color="#8e44ad")
plt.xlabel("Time Cycle", fontsize=12)
plt.ylabel("Remaining Useful Life (RUL)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

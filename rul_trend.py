from config import file_path, plt, sns, pd, df_train

# Compute RUL first
df_train['RUL'] = df_train.groupby(0)[1].transform(max) - df_train[1]

# Select a specific unit (e.g., Unit ID = 1)
unit_id = 1
unit_data = df_train[df_train[0] == unit_id]

# Print to verify RUL exists
print(df_train[['RUL']].head())

# Plot RUL over time (cycle number)
plt.figure(figsize=(10, 5))
plt.plot(unit_data[1], unit_data['RUL'], marker='o', linestyle='-')
plt.xlabel("Time Cycle")
plt.ylabel("Remaining Useful Life (RUL)")
plt.title(f"RUL Trend for Unit {unit_id}")
plt.grid()
plt.show()

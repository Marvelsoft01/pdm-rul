import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed training data with RUL
df_train = pd.read_csv("dataset/df_train_with_rul.csv")

# Now everything else will work
if 'RUL' not in df_train.columns:
    raise ValueError("❌ RUL column is missing! Ensure compute_rul function was applied.")

# Drop non-numeric columns before correlation
correlations = df_train.drop(columns=['unit', 'cycle']).corr()

# Extract correlation with RUL
rul_corr = correlations['RUL'].drop('RUL').sort_values(key=abs, ascending=False)

# Visualization
plt.figure(figsize=(10,6))
sns.barplot(x=rul_corr.values, y=rul_corr.index, palette='coolwarm')
plt.title('Correlation of Sensors with RUL')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

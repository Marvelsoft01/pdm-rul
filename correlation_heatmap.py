
from config import plt , sns , pd
from preprocessing import df_train

# Compute the correlation matrix for the dataset
corr_matrix = df_train.corr()  

# Create a new figure for the heatmap with a specified size
plt.figure(figsize=(12, 8))  

# Generate a heatmap to visualize correlations
sns.heatmap(
    corr_matrix,     # Use the computed correlation matrix
    annot=False,     # Do not display numerical values in the cells
    cmap="coolwarm", # Use the "coolwarm" colormap (blue to red)
    linewidths=0.5   # Set the width of lines separating cells
)

# Set the title of the heatmap plot
plt.title("Feature Correlation Matrix")  

# Display the plot
plt.show().show()
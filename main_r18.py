import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Create a jointplot for RM and PRICE with added alpha for opacity
sns.jointplot(x='RM', y='PRICE', data=data, kind='reg', height=8, joint_kws={'scatter_kws': {'alpha': 0.5}})

# Add a title to the plot
plt.suptitle('Joint Plot of Average Number of Rooms (RM) vs Home Prices (PRICE)', y=1.03)

# Save the plot as a file
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\rm_price_jointplot_with_alpha.png')

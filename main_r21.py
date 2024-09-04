import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Define the features (X) and the target (y)
X = data.drop(columns=['PRICE'])  # All columns except 'PRICE' are features
y = data['PRICE']  # 'PRICE' is the target variable

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Extract the coefficients and create a DataFrame
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# Sort the coefficients by their absolute value for better readability
coef_df.sort_values(by='Coefficient', ascending=False, inplace=True)

# Display the coefficients
print(coef_df)

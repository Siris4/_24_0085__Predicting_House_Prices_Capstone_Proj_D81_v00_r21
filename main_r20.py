import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Define the features (X) and the target (y)
X = data.drop(columns=['PRICE'])  # All columns except 'PRICE' are features
y = data['PRICE']  # 'PRICE' is the target variable

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target values for the training data
y_train_pred = model.predict(X_train)

# Calculate the R-squared value for the training data
r2_train = r2_score(y_train, y_train_pred)

# Output the R-squared value
print(f"R-squared value on training data: {r2_train:.4f}")  # Expecting a number between 0 and 1

# Additional debugging information
print("First 5 actual prices:", y_train.head().tolist())
print("First 5 predicted prices:", y_train_pred[:5].tolist())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
house_data = pd.read_csv("housepredictor.csv")

# Remove duplicate entries
house_data.drop_duplicates(inplace=True)

# Drop rows with any empty entries
house_data.dropna(inplace=True)

# Select relevant columns
house_data = house_data[["price", "area", "bedrooms", "bathrooms", "stories"]]

# Extract columns
area_sqft = house_data["area"]
num_rooms = house_data["bedrooms"]
bathrooms = house_data["bathrooms"]
price = house_data["price"]
stories = house_data["stories"]

# Creating the DataFrame
house_data = pd.DataFrame({
    'Area (sq.ft.)': area_sqft,
    'Number of Rooms': num_rooms,
    'Number of bathrooms': bathrooms,
    'Number of stories': stories,
    'Price ($)': price
})

# Preparing the features (X) and target (y)
X = house_data[['Area (sq.ft.)', 'Number of Rooms', 'Number of bathrooms', 'Number of stories']]
y = house_data['Price ($)']

# Splitting the dataset into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Predicting house prices on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color="blue", label="Predicted vs Actual")
sns.lineplot(x=y_test, y=y_test, color="red", label="Ideal Fit (y=x)", linestyle="-")
plt.xlabel("Actual Prices ($)")
plt.ylabel("Predicted Prices ($)")
plt.title(f"Regression Model: Predicted vs Actual Prices\nRMSE: {rmse:.2f}")
plt.legend()
plt.show()

house_data['Predicted Price ($)'] = model.predict(X)

# Setting up a 2x2 plot grid for each feature against the actual price
plt.figure(figsize=(15, 10))

# Plotting Area (sq.ft.) vs Price
plt.subplot(2, 2, 1)
sns.regplot(x='Area (sq.ft.)', y='Price ($)', data=house_data, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
plt.title("Area (sq.ft.) vs Price ($)")

# Plotting Number of Rooms vs Price
plt.subplot(2, 2, 2)
sns.regplot(x='Number of Rooms', y='Price ($)', data=house_data, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
plt.title("Number of Rooms vs Price ($)")

# Plotting Number of bathrooms vs Price
plt.subplot(2, 2, 3)
sns.regplot(x='Number of bathrooms', y='Price ($)', data=house_data, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
plt.title("Number of Bathrooms vs Price ($)")

# Plotting Number of stories vs Price
plt.subplot(2, 2, 4)
sns.regplot(x='Number of stories', y='Price ($)', data=house_data, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
plt.title("Number of Stories vs Price ($)")

plt.tight_layout()
plt.show()


# Displaying the model's performance
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Model Coefficients (Area, Number of Rooms): {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Optional: Predicting the price of a specific house (example)
example_house = pd.DataFrame({'Area (sq.ft.)': [3000], 'Number of Rooms': [5]})
predicted_price = model.predict(example_house)
print(f"Predicted Price for a house with 3000 sq.ft. and 5 rooms: ${predicted_price[0]:,.2f}")

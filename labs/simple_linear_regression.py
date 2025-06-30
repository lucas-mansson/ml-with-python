# A lab/practice in simple linear regression, predicting CO2-emissions based on features of a car
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Read the data from url
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Look at the complete data
sample1 = df.sample(n=5)
data_describe = df.describe()

# Filter data to what might be relevant for CO2 emissions
cdf = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
filtered_data_sample = cdf.sample(n=9)

# Visualize the features
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
#plot.show() # I am using nvim in gnome terminal, hence this doesnt work and i need to output visualizations to a file
plot.savefig("visualizations/lab1/features.png") 
plot.clf() # Clear figure before next

# Plot fuel consumption against CO2 emissions
plot.scatter(x=cdf.FUELCONSUMPTION_COMB, y=cdf.CO2EMISSIONS, color="blue")
plot.xlabel("Fuel consumption")
plot.ylabel("Emission")
plot.savefig("visualizations/lab1/fuelcons_co2.png") 
plot.clf() 

# Plot engine size vs CO2 emissions
plot.scatter(x=cdf.ENGINESIZE, y=cdf.CO2EMISSIONS, color="blue")
plot.xlabel("Engine size")
plot.ylabel("Emission")
plot.xlim(0, 27)
plot.savefig("visualizations/lab1/enginesize_emissions.png") 
plot.clf() 

# Plot cylinder against CO2
plot.scatter(x=cdf.CYLINDERS, y=cdf.CO2EMISSIONS, color="blue")
plot.xlabel("Number of cylinders")
plot.ylabel("CO2 Emission")
plot.savefig("visualizations/lab1/nbrcylinders_emissions.png") 
plot.clf() 

# Extract input feature and labels from dataset
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# Create test and training sets for inputs and output. 20% of the data becomes test set, 80% for training.
# random_state=42 makes the sets reproducable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(type(X_train), np.shape(X_train), np.shape(X_train))

# Build linear regression model
regressor = linear_model.LinearRegression()

# Train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)
print("Coefficients: ", regressor.coef_[0]) # With simple linear regression we only have one coefficient
print("Intercept: ", regressor.intercept_)

# Visualize how well the model fit the line to the training data
# The regression line is the line given by y = interceptt + coefficient * x
plot.scatter(X_train, y_train)
plot.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, "-r")
plot.xlabel("Engine size")
plot.ylabel("Emission")
plot.savefig("visualizations/lab1/model_line_fit.png") 
plot.clf() 

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

# Evaluate model
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))

# Plot over test data instead of training dataplt.scatter(X_test, y_test, color="blue")    #ADD CODE
plot.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plot.xlabel("Engine size")
plot.ylabel("Emission")
plot.savefig("visualizations/lab1/plot_test_data.png") 
plot.clf() 

# Now create a model to predict based on fuel consumption
X = cdf.FUELCONSUMPTION_COMB.to_numpy() # ADD CODE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regr = linear_model.LinearRegression() 
regr.fit(X_train.reshape(-1, 1), y_train)

y_test_ = regr.predict(X_test.reshape(-1,1))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))

# imports
import pandas as pd
import statsmodels.api as sm


from sklearn.linear_model import LassoCV, RidgeCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#read in data
name = "renamed.csv" # fill in the blank
df = pd.read_csv(name)

if('Timestamp' in df.columns):
    df = df.drop(columns=['Timestamp']) # drop timestamp column if it exists




df = pd.get_dummies(df, drop_first=True, dtype=float) # convert categorical variables to dummy variables
#perform OLS
X = df.drop(columns=['score']) # Independent variables
y = df['score'] # target

X = sm.add_constant(X) # our intercept (beta_0)

model = sm.OLS(y, X).fit() # Fit OLS regression

print("OLS Regression Results:") # OLS Regression Results
print(model.summary())



#perform Lasso
X = df.drop(columns=['score']) # Independent variables
y = df['score'] # target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2026) # train/split
scaler = StandardScaler() # scale data
X_train = scaler.fit_transform(X_train) # fit transform train data
X_test = scaler.transform(X_test) # transform test data


model = LassoCV(cv=5, random_state=2026) # create Lasso model, splits into 5 cross validation folds to find the best alpha
model.fit(X_train, y_train) # fit model
y_pred = model.predict(X_test) # predict test data

#results
print("Best alpha:", model.alpha_)
print("Lasso Regression Results:") # Lasso Regression Results
print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) # Mean Squ
print("R-squared:", r2_score(y_test, y_pred)) # R-squared
print("Coefficients:", model.coef_) # Coefficients
print("Intercept:", model.intercept_) # Intercept







#perform Ridge
X = df.drop(columns=['score']) # Independent variables
y = df['score'] # target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2026) # train/split
scaler = StandardScaler() # scale data
X_train = scaler.fit_transform(X_train) # fit transform train data
X_test = scaler.transform(X_test) # transform test data

model = RidgeCV(cv=5) # create Ridge model, splits into 5 cross validation folds to find the best alpha
model.fit(X_train, y_train) # fit model

#results


print("Ridge Regression Results:") # Ridge Regression Results
print("Best alpha:", model.alpha_)
print("Mean Squared Error:", mean_squared_error(y_test, model.predict(X_test))) # Mean Squared Error
print("R-squared:", r2_score(y_test, model.predict(X_test))) # R-squared
print("Coefficients:", model.coef_) # Coefficients




#perform Elastic Net
X = df.drop(columns=['score']) # Independent variables
y = df['score'] # target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2026) # train/split
scaler = StandardScaler() # scale data
X_train = scaler.fit_transform(X_train) # fit transform train data
X_test = scaler.transform(X_test) # transform test data

model = ElasticNet(alpha = 1, l1_ratio = 0.5, random_state=2026) # create Elastic Net model
model.fit(X_train, y_train) # fit model

y_pred = model.predict(X_test) # predict test data

#results
print("Elastic Net Regression Results:") # Elastic Net Regression Results
print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) # Mean Squ
print("R-squared:", r2_score(y_test, y_pred)) # R-squared
print("Coefficients:", model.coef_) # Coefficients
print("Intercept:", model.intercept_) # Intercept



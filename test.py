from MultivariateLR import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('USA_Housing.csv')
# dropping the address column as it is a string
data.drop('Address', axis=1, inplace=True)
X = data[['Avg. Area Income','Avg. Area Number of Rooms']]
Y = data[['Price']]
# doing train test split for getting training and evaluation samples
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25)

# by default LR is 0.01 and iterations are 1000
# we can pass the values of LR and iterations in the model
lr = LinearRegression(LR=0.01, iterations=1100)

lr.fit(X_train, y_train)
pred = lr.predict(X_test)

score = lr.score(pred, y_test)
print(score)
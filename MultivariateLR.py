import numpy as np

class LinearRegression:
    
    def __init__(self, LR=0.01, iterations=1000):
        self.theta = None
        self.LR = LR
        self.itr = iterations
    
    # function to scale the data between -1 and 1
    def scaleData(self,X):
        j = 0
        arr = np.zeros(X.shape)
        for i in X.columns:
            mean = 0
            temp = X[i]
            mean = np.mean(temp)
            temp =  (temp - mean) / np.std(temp)
            arr[:,j] = temp
            j += 1
        return arr
    
    # function to calculate the loss
    def loss(self, X, Y):
        loss = (np.dot(X, self.theta) - Y)**2
        return np.sum(loss) / (2* float(len(X)))
    
    # minimizing the loss using gradient descent
    def gradientDescent(self, X, Y):
        n = len(X)
        for i in range(self.itr):
            self.theta -= (self.LR / n) * np.sum(X * (np.dot(X, self.theta) - Y))
    
    # fit function to initialize the data
    def fit(self, X, Y):
        print('Scalling the Data')
        X = self.scaleData(X)
        ones = np.ones([X.shape[0],1])
        X = np.concatenate((ones,X),axis=1)
        Y = self.scaleData(Y)
        print('Scalling Done')
        
        # generating the theta values, initially set to 0....
        self.theta = np.zeros([X.shape[1],1])
        
        # initial loss
        print('Initial loss is {}'.format(self.loss(X,Y)))
        
        # performinig gradient descent
        print('Performing Gradinet Descent to minimize the loss')
        self.gradientDescent(X,Y)
        print('Gradient Descent Completed')
        
        print('Loss After Gradient Descent is {}'.format(self.loss(X,Y)))
    
    # function to predict
    def predict(self, X_test):
        X_test = self.scaleData(X_test)
        ones = np.ones([X_test.shape[0],1])
        X_test = np.concatenate((ones,X_test),axis=1)
        pred = np.dot(X_test,self.theta)
        return pred
    
    # evaluate the model using the r square method
    def score(self, pred, y_test):
        y_test = self.scaleData(y_test)
        mean = np.mean(y_test)
        actual = np.sum((y_test - mean)**2)
        estimated = np.sum((pred - mean)**2)
        rsq = 1 - (estimated/actual)
        return rsq
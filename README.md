# Implemention of MultiVariate Linear Regression from scratch using just numpy

Multivarite LR is used when we want to use multiple features in our model.

Multiple features makes our predictions more precise. With multiple feature we will have more deep 
knowledge of our data.

# Hypothesis Used here is:

![alt text][hyp]

[hyp]: https://cdn-images-1.medium.com/max/1650/1*CySZafLwRhQI2SEljpaLlg.png

x0 is always 1.

Also, the above hypothesis can be re-framed in terms of Vector Algebra too:

![alt text][hypp]

[hypp]: https://cdn-images-1.medium.com/max/1925/1*_eZxSV8XLI_N2iTG6gESdA.png


Feature Scaling is also used in this to make calculations eaiser and to avoid buffer overflow 
as lot of numerical operations is performed in gradient descent method.

# Feature Scaling:

![alt text][fs]

[fs]: https://cdn-images-1.medium.com/max/1100/1*0kO2HpKGl_B1UVkNMSmueQ.png

where u is the Mean and sigma is the Standard Deviation:
![alt text][ms]

[ms]: https://cdn-images-1.medium.com/max/1100/1*DqHqxjFBQEB0P08Cp5y2Aw.png


Gradient Descent method is used to minimize the loss or the cost funtion.

# Gradient Descent in Multi Variate LR:
![alt text][gd]

[gd]: https://raw.githubusercontent.com/ritchieng/machine-learning-stanford/master/w2_linear_regression_multiple/multivariate_algo.png



Initial value of Learning Rate is 0.01 and iterations are set to 1000.

You can also specify these values while creating the instance or object of the LinearRegression class.

A test file is also included with a dataset which shows how to use the model.

A jupyter Notebook is of the model is also included with full implementation and testing.
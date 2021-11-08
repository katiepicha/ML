'''
MACHINE LEARNING

What is Machine Learning?
- subfield of AI
- uses algorithms and statistical models to predict outcome
- relies on patterns and makes inferences
- uses historical data as input to predict new output

Scikit-Learn Library
- packages the most effective machine-learning algorithms (estimators)
- create models for data analysis, extracting insights and making predictions
    - trains your model
    - test your model
    - put it to work (make predictions)

Types of ML
- Supervised ML: works with labeled data
    - training your model to recognize objects that are labeled so it can classify the object
- Unsupervised ML: works with unlabeled data
    - putting your model to work to classify objects that are not labeled based on the training under the supervised approach

Supervised ML
- train your ML models on datasets that consists of rows and columns
    - rows = data sample
        - each sample has an associated label called a target: the value that you want your model to predict
    - columns = feature of that sample
- 2 categories of Supervised ML
    1) Classification model: predict the discrete classes (categories) to which samples belong to
        - binary classification = uses just 2 classes, it will fall into one or the other
        - multi-classification = uses more than 2 classes
    2) Regression model: predict a continuous output
        - LinearRegression estimator = use to perform simple linear regression or multiple linear regression

Unsupervised ML
- clustering algorithm = the goal of this algorithm is to find similarities in the datapoint and group similar datapoints together

Steps in a Typical Data Science Study
- loading the dataset
- exploring the data with pandas and visualizations
- transforming your data
- splitting the data for training and testing
- creating the model
- training and testing the model
- tuning the model and evaluating its accuracy
- making predictions on live data that the model hasn't seen before

Simple Linear Regression
- regression line = single line that best fits the data (in terms of having the smallest overall distance from the line to the points)

'''
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

<<<<<<< HEAD
Simple Linear Regression
- regression line = single line that best fits the data (in terms of having the smallest overall distance from the line to the points)
=======
DIGITS DATASET
- digits is a bunch object
    - a bunch is a dictionary with additional dataset-specific attributes
    - the bunch object has 3 different attributes
        1) data - contains all of our samples
        2) target - the images' labels, (classes) indicating which digit each image represents
        3) images - a Numpy array of type float that represents the pixel intensity for each pixel in an 8-by-8 image

Splitting the Data for Training and Testing
- Typically train a model with a subset of a dataset
- Save a portion for testing, so you can evaluate a model’s performance using unseen data
- Function train_test_split shuffles the data to randomize it, then splits the samples in the data array 
  and the target values in the target array into training and testing sets
- Shuffling helps ensure that the training and testing sets have similar characteristics
- By default, train_test_split reserves 75% of the data for training and 25% for testing

Creating the Model
- In scikit-learn, models are called estimators
    - KNeighborsClassifier estimator implements the k-nearest neighbors algorithm
- Train the model using the ‘fit’ method
    - A ‘fit’ method of a model typically loads the data into the model and performs complex calculations. 
      The KNeighborsClassifier fit method just loads the data but does not perform any calculations (training)

Confusion Matrix
- shows correct and incorrect predicted values (the hits and misses) for a given class
- correct predictions are shown on the principle diagonal from top left to bottom right
- nonzero values not on the principle diagonal indicate incorrect predictions
- each row represents one distinct class
- columns specify how many test samples were classified into classes
>>>>>>> bdc5d4647891666bec895d9518e82495c0357d4c

'''
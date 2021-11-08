from sklearn.datasets import load_digits

# digits is our dataset and is referred to as a bunch object
# a bunch is a dictionary with additional dataset-specific attributes
digits = load_digits()

# returns text about the digits dataset as well as the number of instances and attributes
print(digits.DESCR)

# data attribute contains all of our samples
# the 1797 samples (digit images), each with 64 features with values 0 (white) to 16 (black), representing pixel intensities
print(digits.data[150])
# target attribute is the images' labels, (classes) indicating which digit each image represents
print(digits.target[150])

print(digits.data[5])
print(digits.target[5])

print(digits.data.shape) # data is (1797, 64)
print(digits.target.shape) # target is just (1797,)


import matplotlib.pyplot as plt

# creates a chart of 24 pictures
figure, axes = plt.subplots(nrows = 4, ncols = 6, figsize = (6,4))

# plt.show()

# iterates through the subplots, images, and target at the same time
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.imshow(image, cmap = plt.cm.gray_r) # displays grayscale image
    axes.set_xticks([]) # removes x-axis tick marks
    axes.set_yticks([]) # removes y-axis tick marks
    axes.set_title(target) # titles with the target value of the image

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(digits.data, digits.target, random_state = 11)

print(data_train.shape) # data is two dimensional
print(data_test.shape)
print(target_train.shape) # target is one dimensional
print(target_test.shape)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# fit performs all of the machine learning - requires the data and the target
knn.fit(X = data_train, y = target_train)

predicted = knn.predict(X = data_test) # does not need y because it will be making the predictions
expected = target_test

print(predicted[:20])
print(expected[:20])

# score will produce a score based on how well it was predicted - how accurate it was
print(format(knn.score(data_test, target_test), ".2%"))

# iterating through both the predicted and expected at the same time to compare the two
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)


from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true = expected, y_pred = predicted)
print(confusion)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index = range(10), columns = range(10))

figure = plt2.figure(figsize = (7,6))
axes = sns.heatmap(confusion_df, annot = True, cmap = plt2.cm.nipy_spectral_r) # higher the number, the darker the scale

plt2.show()
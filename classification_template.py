# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv', encoding="cp949")         
#광고에 반응 여부

X = dataset.iloc[:, [2, 3]].values
#[a:b] -- a <= x < b in Python
#[a:b] -- a <= x <= b in iloc in pandas.read_csv 
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# compute mean and std of X_train - 300개 데이터
# new_X_train = (old_X_train - mean) / std
X_test = sc.transform(X_test)
# 방금전에 구한 mean, std을 이용하여 스케일을 보정하라. 

# Training the Logistic Regression model on the Training set - Classifier is needed here
from sklearn.linear_model import LogisticRegression         
# Class
classifier = LogisticRegression(random_state = 0, solver='lbfgs')
classifier.fit(X_train, y_train)                            
# 주어진 데이터에 맞는 모델계수 계산 

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix 
# (오차행렬: 예측과 실제를 비교하기 위한 표) 
# To Evaluate Prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap               # class
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
            # X1.shape = (592, 616), X1.ravel().shape = (364672,)
            # np.array([X1.ravel(), X2.ravel()]).T - transpose, numpy.ndarray.ravel(): return flatted array
            # classifier.predict(np.array([X1.ravel(), X2.ravel()]).T) --> 0 or 1 depending on np.array([X1.ravel(), X2.ravel()]).T 
            # alpha = 0 - transparent, 1 - opaque
            # classifier.predict(...).reshape(X1.shape) -- Z: The height values over which the contour is drawn
                        
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):        # unique한 것들로만 만듬: enumerate --> (0, a[0]), (1, a[1]), ...: a[0]=0, a[1]=1 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
    # label = 0인 것은 red, label = 1인 것은 green으로 그려라. 
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
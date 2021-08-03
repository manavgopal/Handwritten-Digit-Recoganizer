import numpy as np
import pandas as pd
%matplotlib notebook
import matplotlib.pyplot as plt

train_data = pd.read_csv("../input/prmllab/mnist_train.csv")
test_data = pd.read_csv("../input/prmllab/test.csv")
X_test = test_data.drop(["id"], axis=1)

train_data.head()

test_data

X_test.head()

X = train_data.drop(['5'],axis = 1)
y = train_data[['5']]
X.head()

def plotNum(ind):
    plt.imshow(np.reshape(np.array(data.iloc[ind,1:]), (28, 28)), cmap="gray")

# plt.figure()
# for ii in range(1,17):
#     plt.subplot(4,4,ii)
#     plotNum(ii)

# X = train_data.iloc[:, 1:]
# y = train_data['label']

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# X_train, X_val, y_train, y_val = train_test_split(X, y.values.ravel(),test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1500,criterion="gini",n_jobs=-1,random_state=42,verbose=10)
clf.fit(X,y.values.ravel())

# skf = StratifiedKFold(n_splits=5,random_state=5,shuffle=True)
# skf.get_n_splits(X, y)
# max_acc=0
# print(skf)
# for train_index, test_index in skf.split(X, y.values.ravel()):
#     X_train_, X_test_ = X.iloc[train_index], X.iloc[test_index]
#     y_train_, y_test_ = y.values.ravel()[train_index], y.values.ravel()[test_index]
#     clf.fit(X_train_,y_train_)
#     pred=clf.predict(X_test_)
#     acc=accuracy_score(y_test_,pred)
#     if(acc>max_acc):
#         max_acc=acc
#         X_train_acc=X_train_
#         y_train_acc=y_train_
#         X_test_acc=X_test_
#         y_test_acc=y_test_
# clf.fit(X_train_acc,y_train_acc)

out = clf.predict(X_test)
output = range(0,len(out))
# print(len(output))
data_to_submit = pd.DataFrame({
    'id': output,
    'class': out
})
data_to_submit.to_csv('./submission.csv', index = False)

import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.svm import SVC
import time

def main():
    path='train_data.csv'
    rawData= np.loadtxt(path,delimiter=',')

    n_feature=512
    x,y = np.split(rawData, (n_feature,), axis=1);
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8);
    start = time.time()
    clf = SVC(kernel='rbf', C=10, gamma=0.1);
    clf.fit(x_train,y_train.ravel())
    end = time.time()

    print("Accuracy on training set: {:.2f}".format(clf.score(x_train, y_train)))
    print("Running time: {:.2f} ms".format((end - start) * 1000))
    test_start = time.time()
    print("Accuracy on test set: {:.2f}".format(clf.score(x_test, y_test)))
    test_end = time.time()
    print("Test Running time: {:.2f} ms".format((test_end - test_start) * 1000))

if __name__ == '__main__':
    main();
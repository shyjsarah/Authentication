import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time


def main():
    path='train_data.csv'
    rawData= np.loadtxt(path,delimiter=',')

    x,y = np.split(rawData, (512,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8);
    #训练阶段
    #pca

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    n_feature=50
    pca = PCA(n_components=n_feature)
    pca.fit(x_train_scaled)
    x_train_pca = pca.transform(x_train_scaled)
    print("Original shape: {}".format(str(x_train.shape)))
    print("Reduced shape: {}".format(str(x_train_pca.shape)))

    #svm
    svm_start = time.time()
    clf = SVC(kernel='rbf', C=10, gamma=0.01);
    clf.fit(x_train,y_train.ravel())
    svm_end = time.time()
    print("Accuracy on training set: {:.2f}".format(clf.score(x_train, y_train)))
    print("Running time: {:.2f} ms".format((svm_end-svm_start)*1000));

    x_test_scaled = scaler.transform(x_test)
    x_test_pca = pca.transform(x_test_scaled)
    print("Original shape: {}".format(str(x_test.shape)))
    print("Reduced shape: {}".format(str(x_test_pca.shape)))
    print("Accuracy on test set: {:.2f}".format(clf.score(x_test_pca, y_test)))


if __name__ == '__main__':
    main();
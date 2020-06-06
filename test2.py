import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time


def main():
    path = 'train_data.csv'
    rawData = np.loadtxt(path, delimiter=',')

    x, y = np.split(rawData, (512,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8);
    # 训练阶段
    # pca
    pca_start = time.time()
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    n_feature = 50
    pca = PCA(n_components=n_feature)
    pca.fit(x_train_scaled)
    x_train_pca = pca.transform(x_train_scaled)
    pca_end = time.time()
    print("Original shape: {}".format(str(x_train.shape)))
    print("Reduced shape: {}".format(str(x_train_pca.shape)))
    print("PCA Running time: {:.2f} ms".format((pca_end - pca_start) * 1000));
    # svm
    svm_start = time.time()
    clf = SVC(kernel='rbf', C=10, gamma=0.01);
    clf.fit(x_train_pca, y_train.ravel())
    svm_end = time.time()
    print("Accuracy on training set: {:.2f}".format(clf.score(x_train_pca, y_train)))
    print("SVM Running time: {:.2f} ms".format((svm_end - svm_start) * 1000));

    test_start=time.time()
    x_test_scaled = scaler.transform(x_test)
    x_test_pca = pca.transform(x_test_scaled)
    print("Original shape: {}".format(str(x_test.shape)))
    print("Reduced shape: {}".format(str(x_test_pca.shape)))
    print("Accuracy on test set: {:.2f}".format(clf.score(x_test_pca, y_test)))
    test_end = time.time()
    print("Test Running time: {:.2f} ms".format((test_end - test_start) * 1000))


if __name__ == '__main__':
    main();
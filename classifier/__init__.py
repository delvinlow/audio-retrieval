__author__ = 'xiangwang1223@gmail.com'

import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.metrics import classification_report

# Trian your own classifier.
# Here is the simple implementation of SVM classification.
def mySVM(mat_path, output_path):
    # 1. Load matricies from the .mat file, including X_train, X_test, Y_train, Y_gnd;
    mat_contents = sio.loadmat(mat_path)

    # 2. X_train with instance_num * feature_num dimensions, and its corresponding venue labels Y_train
    #    with instance_num * class_num dimensions.
    #    As well as X_test, and its label matrix Y_gnd.
    X_train = np.asmatrix(mat_contents['X_train'])
    Y_train = np.asmatrix(mat_contents['Y_train'])

    X_test  = np.asmatrix(mat_contents['X_test'])
    Y_gnd  = np.asmatrix(mat_contents['Y_gnd'])
    print('Data Load Done.')

    # 3. Generate the predicted label matrix Y_predicted for X_test via SVM or other classifiers.
    instance_num, class_num = Y_gnd.shape

    Y_predicted = np.asmatrix(np.zeros([instance_num, class_num]))

    # 4. Train the classifier.
    model = svm.SVR(kernel='rbf', degree=3, gamma=0.1, shrinking=True, verbose=False, max_iter=-1)
    model.fit(X_train, Y_train)
    Y_predicted = np.asmatrix(model.predict(X_test))
    print('SVM Train Done.')

    # 5. Save the predicted results and ground truth.
    sio.savemat(output_path, {'Y_predicted': Y_predicted, 'Y_gnd': Y_gnd})


if __name__ == '__main__':
    mat_path = 'Sample.mat'
    output_path = 'Output.mat'

    mySVM(mat_path, output_path)
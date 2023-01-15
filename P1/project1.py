# EECS 445 - Winter 2018
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """

    if penalty == 'l1': return LinearSVC(penalty = 'l1', dual = False, C = c, class_weight = class_weight, max_iter = 10000)
    if degree == 1: return SVC(kernel='linear', C=c, class_weight=class_weight, degree = degree)
    if degree == 2: return SVC(gamma = 'auto', kernel='poly', C=c, class_weight=class_weight, degree = degree, coef0 = r)
    # TODO: Optionally implement this helper function if you would like to
    # instantiate your SVM classifiers in a single function. You will need
    # to use the above parameters throughout the assignment.


def extract_dictionary(df):
    """
    Reads a panda dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df
    """
    word_dict = {}
    # TODO: Implement this function
    set_d = set()
    for i in range(df.index.size):
        str = df.loc[i]["text"].lower()
        for c in string.punctuation:
            str = str.replace(c, ' ')
        set_d = set_d.union(set(str.split()))
    set_d = list(set_d)
    for i in range(len(set_d)):
        word_dict[set_d[i]] = i
    return word_dict


def generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (number of reviews, number of words).
    Input:
        df: dataframe that has the ratings and labels
        word_list: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (number of reviews, number of words)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # TODO: Implement this function
    for i in range(df.index.size):
        str = df.loc[i]["text"].lower()
        for c in string.punctuation:
            str = str.replace(c, ' ')
        for part in str.split():
            if part in word_dict: feature_matrix[i][word_dict.get(part)] = 1
    return feature_matrix


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    #HINT: You may find the StratifiedKFold from sklearn.model_selection
    #to be useful

    #Put the performance of the model on each fold in the scores array
    scores = []
    skf = StratifiedKFold(k)
    skf.get_n_splits(X, y)

    for train_ind, test_ind in skf.split(X, y):
        X_train = X[train_ind]
        y_train = y[train_ind]
        clf.fit(X_train,y_train)
        X_test = X[test_ind]
        if metric == 'AUROC':
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)
        y_true = y[test_ind]
        scores.append(performance(y_true, y_pred, metric))

    #And return the average performance across all fold splits.
    return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
    Returns:
        The parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    #HINT: You should be using your cv_performance function here
    #to evaluate the performance of each SVM
    
    
    max, max_val = 0, 0
    for potential in C_range:
        clf = select_classifier(c = potential, penalty = penalty)
        cur = cv_performance(clf,X,y,k,metric)
        if cur > max_val:
            max = potential
            max_val = cur
        print(potential, ":", cur)
    print(metric, ":", max, max_val)
    return max


def plot_weight(X,y,penalty,metric,C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []

    # TODO: Implement this part of the function
    #Here, for each value of c in C_range, you should
    #append to norm0 the L0-norm of the theta vector that is learned
    #when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    for potential in C_range:
        clf = select_classifier(c=potential, penalty = penalty)
        clf.fit(X,y)
        norm_c0 = 0
        for num in clf.coef_[0]:
            if num != 0: norm_c0 += 1
        norm0.append(norm_c0)



    #This code will plot your L0-norm as a function of c
    plt.plot(C_range, norm0)
    plt.xscale('log')
    plt.legend(['L0-norm'])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title('Norm-'+penalty+'_penalty.png')
    plt.savefig('Norm-'+penalty+'_penalty.png')
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """
        Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
        calculating the k-fold CV performance for each setting on X, y.
        Input:
            X: (n,d) array of feature vectors, where n is the number of examples
               and d is the number of features
            y: (n,) array of binary labels {1,-1}
            k: an int specifying the number of folds (default=5)
            metric: string specifying the performance metric (default='accuracy'
                     other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                     and 'specificity')
            parameter_values: a (num_param, 2)-sized array containing the
                parameter values to search over. The first column should
                represent the values for C, and the second column should
                represent the values for r. Each row of this array thus
                represents a pair of parameters to be tried together.
        Returns:
            The parameter value(s) for a quadratic-kernel SVM that maximize
            the average 5-fold CV performance
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...

    max_c,max_r, max_val = 0, 0, 0
    for potent_c, potent_r in param_range:
        clf = select_classifier(c = potent_c, r = potent_r, degree = 2)
        cur = cv_performance(clf,X,y,k,metric)
        if cur > max_val:
            max_c = potent_c
            max_r = potent_r
            max_val = cur
    print(metric, ":", max_c, max_r, max_val)
    return [max_c, max_r]


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels y_pred.
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    # TODO: Implement this function
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
        # Accuracy = (FP + FN) / N
    if (metric == 'Accuracy'):
        return metrics.accuracy_score(y_true, y_pred)
    
    # Recall/Sensitivity = TP / (TP + FN)
    elif (metric == 'Sensitivity'):
        return metrics.recall_score(y_true, y_pred)
    
    # Precision = TP / (TP + FP)
    elif (metric == 'Precision'):
        return metrics.precision_score(y_true, y_pred)
    
    # F1-Score = 2 * Precision * Sensitivity / (Precision + Sensitivity)
    elif (metric == "F1-Score"):
        return metrics.f1_score(y_true, y_pred)
    
    # AUROC
    elif (metric == "AUROC"):
        return metrics.roc_auc_score(y_true, y_pred)

    #Specificity = TN / (TN + FP)
    elif (metric == "Specificity"):
        TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
        return TN / (TN + FP)







def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()

    # 2
    # print("Second Part:")
    # print("d =",X_train.shape[1])
    # count = 0
    # for i in range(X_train.shape[0]):
    #     count += np.count_nonzero(X_train[i])
    # print("AVE =", count/X_train.shape[0])

    # print("Third Part:")
    # 3.1 (c)
    # print('(c)')
    # C_range = [10 ** x for x in range(-3, 4)]
    # select_param_linear(X_train,Y_train,5,"Accuracy", C_range)
    # select_param_linear(X_train,Y_train,5,"F1-Score", C_range)
    # select_param_linear(X_train,Y_train,5,"AUROC", C_range)
    # select_param_linear(X_train,Y_train,5,"Precision", C_range)
    # select_param_linear(X_train,Y_train,5,"Sensitivity", C_range)
    # select_param_linear(X_train,Y_train,5,"Specificity", C_range)

    # 3.1 (d)
    # clf = select_classifier(c = 0.1)
    # clf.fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)
    # Y_pred_AU = clf.decision_function(X_test)
    # print('(d)')
    # print("Accuray =", performance(Y_test, Y_pred,'Accuracy'))
    # print("F1-Score =", performance(Y_test, Y_pred,'F1-Score'))
    # print("AUROC =", performance(Y_test, Y_pred_AU, 'AUROC'))
    # print("Precision =", performance(Y_test, Y_pred, 'Precision'))
    # print("Sensitivity =", performance(Y_test, Y_pred, 'Sensitivity'))
    # print("Specificity =", performance(Y_test, Y_pred, 'Specificity'))

    # print('(e)')
    # plot_weight(X_train,Y_train,'L2','Accuracy', C_range)

    # print('(f)')
    # clf = select_classifier(c = 0.1)
    # clf.fit(X_train,Y_train)
    # coef = np.array(clf.coef_[0])
    # large_4 = np.argpartition(coef, -4)[-4:]
    # least_4 = np.argpartition(coef, 4)[:4]
    # for key, value in dictionary_binary.items():
    #     for item in large_4:
    #         if value == item:
    #             print(key, coef[value])
    #     for item in least_4:
    #         if value == item:
    #             print(key, coef[value])

    # 3.2 (b)
    # Grid Search
    # print('3.2 (b)')
    # param_range = []
    # for i in range(-3,4):
    #     for j in range (-3,4): param_range.append([10**i, 10**j])
    # select_param_quadratic(X_train, Y_train, 5, "AUROC", param_range)

    # Random Search
    # param_range = []
    # c = np.random.uniform(-3,3,25)
    # r = np.random.uniform(-3,3,25)
    # for i in range(25): param_range.append([10 ** c[i],10 ** r[i]])
    # select_param_quadratic(X_train, Y_train, 5, "AUROC", param_range)

    # 3.4 (a)
    # select_param_linear(X_train, Y_train, 5, 'AUROC', C_range, 'l1')

    # 3.4 (b)
    # plot_weight(X_train, Y_train, 'l1', 'AUROC', C_range)

    # 4.1 (b)
    # clf = select_classifier(c = 0.01, class_weight = {-1:10, 1:1})
    # clf.fit(X_train, Y_train)
    # Y_pred = clf.predict(X_test)
    # Y_pred_AU = clf.decision_function(X_test)
    # print("Accuray =", performance(Y_test, Y_pred,'Accuracy'))
    # print("F1-Score =", performance(Y_test, Y_pred,'F1-Score'))
    # print("AUROC =", performance(Y_test, Y_pred_AU, 'AUROC'))
    # print("Precision =", performance(Y_test, Y_pred, 'Precision'))
    # print("Sensitivity =", performance(Y_test, Y_pred, 'Sensitivity'))
    # print("Specificity =", performance(Y_test, Y_pred, 'Specificity'))



    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # 4.2
    # clf = select_classifier(c = 0.01, class_weight = {-1:7, 1:3})
    # clf.fit(IMB_features, IMB_labels)
    # Y_pred = clf.predict(IMB_test_features)
    # Y_pred_AU = clf.decision_function(IMB_test_features)
    # print("Accuray =", performance(IMB_test_labels, Y_pred,'Accuracy'))
    # print("F1-Score =", performance(IMB_test_labels, Y_pred,'F1-Score'))
    # print("AUROC =", performance(IMB_test_labels, Y_pred_AU, 'AUROC'))
    # print("Precision =", performance(IMB_test_labels, Y_pred, 'Precision'))
    # print("Sensitivity =", performance(IMB_test_labels, Y_pred, 'Sensitivity'))
    # print("Specificity =", performance(IMB_test_labels, Y_pred, 'Specificity'))

    # 4.3 (a)
    # max_val = 0
    # max_i = 0;
    # max_j = 0;
    # for i in range(1, 10):
    #     for j in range (i, 10):
    #         clf = select_classifier(c = 0.01, class_weight = {-1:i, 1:j})
    #         perform = cv_performance(clf, IMB_features, IMB_labels, 5, 'F1-Score')
    #         if (max_val < perform) :
    #             max_val = perform
    #             max_i = i
    #             max_j = j
    # print("Max Neg =", max_i)
    # print("Max Pos =", max_j)
    # print("Max AUROC =", max_val)


    # 4.3 (b)
    clf = select_classifier(c = 0.01, class_weight = {-1:1, 1:1.8})
    clf.fit(IMB_features, IMB_labels)
    Y_pred = clf.predict(IMB_test_features)
    Y_pred_AU = clf.decision_function(IMB_test_features)
    print("Accuray =", performance(IMB_test_labels, Y_pred,'Accuracy'))
    print("F1-Score =", performance(IMB_test_labels, Y_pred,'F1-Score'))
    print("AUROC =", performance(IMB_test_labels, Y_pred_AU, 'AUROC'))
    print("Precision =", performance(IMB_test_labels, Y_pred, 'Precision'))
    print("Sensitivity =", performance(IMB_test_labels, Y_pred, 'Sensitivity'))
    print("Specificity =", performance(IMB_test_labels, Y_pred, 'Specificity'))

    # 4.4
    # clf_1 = select_classifier(c = 0.01, class_weight = {-1:1, 1:1})
    # clf_1.fit(IMB_features, IMB_labels)
    # Y_pred_1 = clf_1.predict(IMB_test_features)
    # Y_pred_AU_1 = clf_1.decision_function(IMB_test_features)
    # fp_1, tp_1, thresholds = metrics.roc_curve(IMB_test_labels, Y_pred_AU_1)
    # fp, tp, thresholds = metrics.roc_curve(IMB_test_labels, Y_pred_AU)
    # plt.title('ROC Comparison Curve')
    # plt.plot(fp, tp, '-g', label = 'Wn = 5, Wp = 9')
    # plt.plot(fp_1, tp_1, '-y', label = 'Wn = 1, Wp = 1')
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc = 'lower right')
    # plt.show()



    # TODO: Questions 2, 3, 4

    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    #multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    #heldout_features = get_heldout_reviews(multiclass_dictionary)


if __name__ == '__main__':
    main()

# EECS 445 - Winter 2018
# Project 1 - projct1_solution.py

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *

def extract_word(input_string):
    """
    A helper function that strips input_string of any punctuation by turning
    punctuation into spaces, and then splits along whitespace and returns
    the resulting array.
    """
    for i in string.punctuation:
        input_string = input_string.replace(i, ' ')
    return input_string.lower().split()


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
    for i in df['text']:
        for j in extract_word(i):
            if j not in word_dict:
                word_dict[j] = len(word_dict)
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
    for count, text in enumerate(df['text']):
        for j in extract_word(text):
            if j in word_dict:
                feature_matrix[count][word_dict[j]] = 1
    return feature_matrix


def svm_sign(Y_pred):
    """
    Helper function maps continuous output of SVM to binary labels
    """
    Y_pred=np.sign(Y_pred)
    Y_pred[Y_pred == 0] = 1
    return Y_pred


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
    skf = StratifiedKFold(n_splits=k)
    scores = []
    # For each split in the k folds...
    for train, test in skf.split(X,y):
        X_train, X_test, Y_train, Y_test = X[train], X[test], y[train], y[test]
        # Fit the data to the training data...
        clf.fit(X_train, Y_train)
        # And test on the ith fold.
        if metric == "accuracy":
            Y_pred = clf.predict(X_test)
        else:
            Y_pred = clf.decision_function(X_test)
        score = performance(Y_test, Y_pred, metric)
        if not np.isnan(score):
            scores.append(score)
    # Return the average performance across all fold splits.
    return np.array(scores).mean()


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
    """
    Return a linear svm classifier based on the given
    penalty function and regularization parameter c.
    """
    if degree == 1:
        if penalty == 'l2':
            return SVC(kernel='linear', C=c, class_weight=class_weight)
        elif penalty == 'l1':
            return LinearSVC(penalty='l1', dual=False, C=c, class_weight=class_weight)
    elif degree == 2:
        if penalty == 'l1':
            raise ValueError('Error: a degree 2 SVM with l1 penalty is not supported.')
        return SVC(kernel='poly', degree=2, C=c, coef0=r, class_weight=class_weight)


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
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    print("Linear SVM Hyperparameter Selection based on %s:" %metric)
    scores = []
    # Iterate over all of the given c parameters...
    for c in C_range:
        # Calculate the average performance on k-fold cross-validation
        clf = select_classifier(penalty, c)
        score = cv_performance(clf, X, y, k, metric)
        print("c: %.6f score: %.4f" %(c, score))
        if not np.isnan(score):
            scores.append((c, score))
    # Return the C value with the maximum score
    maxval = max(scores, key=lambda x: x[1])
    return maxval[0]


def plot_weight(X, y, penalty, metric, C_range):
    """
    Takes as input the training data X and labels y and plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier.
    """
    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    norm0 = []
    for c in C_range:
        clf = select_classifier(penalty,c)
        clf.fit(X, y)
        w = clf.coef_
        w = np.squeeze(np.asarray(w))
        norm0.append(np.linalg.norm((w),ord=0))
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
    print("Quadratic SVM Hyperparameter Selection based on %s:" %metric)
    scores = []
    # For each parameter pair to try...
    for p in param_range:
        c = p[0]
        r = p[1]
        clf = select_classifier(c=c, degree=2, r=r)
        # Determine the performance of the defined SVM
        score = cv_performance(clf, X, y, k, metric)
        print("c: %.6f coefficient: %0.6f score: %.4f" % (c, r, score))
        if not np.isnan(score):
            scores.append((r, c, score))
    # And report the pair (C,r) that yielded the best metric performance
    maxval = max(scores, key=lambda x: x[2])
    return maxval[0], maxval[1]


def performance(y_true, Y_pred, metric="accuracy"):
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted labels Y_pred.
    Input:
        y_true: (n,) array containing known labels
        Y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    y_label = svm_sign(Y_pred)
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_label)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_label)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, Y_pred)
    elif metric == "sensitivity":
        conf_mat = metrics.confusion_matrix(y_true, y_label, [1, -1])
        tp = conf_mat[0, 0]
        fn = conf_mat[0, 1]
        return np.float64(tp)/(tp+fn)
    elif metric == "specificity":
        conf_mat = metrics.confusion_matrix(y_true, y_label, [1, -1])
        tn = conf_mat[1, 1]
        fp = conf_mat[1, 0]
        return np.float64(tn)/(tn+fp)


def q2c(X):
    """
    Given a feature matrix X, prints d, the number of unique words found in X,
    and prints the average number of nonzero features per rating in the training data.
    """
    print("Question 2(c): reporting dataset statistics:")
    d = X.shape[1]
    average_num_features = np.mean(np.sum(X, axis=1))
    print("d: " + str(d))
    print("Average number of nonzero features: " + str(average_num_features))


def q3(X_train, Y_train, X_test, Y_test, metric_list, dictionary):
    print("--------------------------------------------")
    ##################################################################
    print("Question 3.1(c): Linear SVM with grid search")
    for metric in metric_list:
        bestc = select_param_linear(X_train, Y_train, 5, metric, 10.0 ** np.arange(-3, 4))
        print("Best c: %.6f" % bestc)

        linear_clf = select_classifier(penalty='l2', c=bestc)
        linear_clf.fit(X_train, Y_train)

        if metric == "accuracy":
            Y_pred = linear_clf.predict(X_test)
        else:
            Y_pred = linear_clf.decision_function(X_test)
        test_perf = performance(Y_test, Y_pred, metric)
        print("Test Performance: %.4f" % test_perf)

    ##################################################################
    print("Question 3.1(d): Performance of SVM with best C on test data")
    cRange = [0.001, 0.01, 0.1]
    for bestc in cRange:  # C value that maximizes accuracy and F1-score performance
        clf = select_classifier(penalty='l2', c=bestc, class_weight={-1: 1, 1: 1})
        clf.fit(X_train, Y_train)
        for metric in metric_list:
            if metric == "accuracy":
                Y_pred = clf.predict(X_test)
            else:
                Y_pred = clf.decision_function(X_test)
            test_perf = performance(Y_test, Y_pred, metric)
            print("C = " + str(bestc) + " Test Performance on metric " + metric + ": %.4f" % test_perf)


    ##################################################################
    print("Question 3.1(e): Plot the weights of C vs. L0-norm of theta")
    plot_weight(X_train, Y_train, 'l2', "auroc", 10.0 ** np.arange(-3, 4))

    ##################################################################
    print("Question 3.1(f): Displaying the most positive and negative words")
    bestc = 0.1
    linear_clf = select_classifier(penalty='l2', c=bestc)
    linear_clf.fit(X_train, Y_train)
    theta = linear_clf.coef_
    theta = np.squeeze(np.asarray(theta))
    max_theta = np.argpartition(theta, -4)[-4:]
    min_theta = np.argpartition(-1.0 * theta, -4)[-4:]
    for index in max_theta:
        for word, word_ind in dictionary.items():
            if index == word_ind:
                print('coeff: ' + '%f' % theta[index] + ' word: ' + str(word))
    for index in min_theta:
        for word, word_ind in dictionary.items():
            if index == word_ind:
                print('coeff: ' + '%f' % theta[index] + ' word: ' + str(word))

    # quadratic SVM
    print("Question 3.2: Quadratic SVM")
    print("3.2(a)i: Quadratic SVM with grid search and auroc metric:")
    C_range = 10.0 ** np.arange(-3, 4)
    coeff_range = 10.0 ** np.arange(-3, 4)
    param_range = np.asarray(list(itertools.product(C_range, coeff_range)))
    bestr, bestc = select_param_quadratic(X_train, Y_train, 5, "auroc", param_range)
    print("Best c: %.6f Best coeff: %.5f" % (bestc, bestr))
    quadratic_clf = select_classifier(c=bestc, degree=2, r=bestr)
    quadratic_clf.fit(X_train, Y_train)
    Y_pred = quadratic_clf.decision_function(X_test)
    test_perf = performance(Y_test, Y_pred, "auroc")
    print("Test Performance: %.4f" % test_perf)

    ##################################################################
    print("3.2(a)ii: Quadratic SVM with random search and auroc metric:")
    a = np.random.uniform(-3.2, 3.2, [25, 1])
    C_range = 10.0 ** a
    a = np.random.uniform(-3.2, 3.2, [25, 1])
    coeff_range = 10.0 ** a
    param_range = np.concatenate((C_range, coeff_range), axis=1)
    bestr, bestc = select_param_quadratic(X_train, Y_train, 5, "auroc", param_range)
    print("Best c: %.6f Best coeff: %.5f" % (bestc, bestr))
    quadratic_clf = select_classifier(c=bestc, degree=2, r=bestr)
    quadratic_clf.fit(X_train, Y_train)
    Y_pred = quadratic_clf.decision_function(X_test)
    test_perf = performance(Y_test, Y_pred, "auroc")
    print("Test Performance: %.4f" % test_perf)

    # linear SVM
    ##################################################################
    print("Question 3.4(a): Linear SVM with l1-penalty, grid search and auroc")

    bestc = select_param_linear(X_train, Y_train, 5, "auroc", 10.0 ** np.arange(-3, 4), 'l1')
    print("Best c: %.6f" % bestc)
    linear_clf_l1 = select_classifier(penalty='l1', c=bestc)
    linear_clf_l1.fit(X_train, Y_train)
    Y_pred = linear_clf_l1.decision_function(X_test)
    test_perf = performance(Y_test, Y_pred, "auroc")
    print("Test Performance: %.4f" % test_perf)

    ##################################################################
    print("Question 3.4(b): Plot the weights of C vs. L0-norm of theta, l1 penalty")
    plot_weight(X_train, Y_train, 'l1', "accuracy", 10.0 ** np.arange(-3, 4))


def q4(X_train, Y_train, X_test, Y_test, IMB_features, IMB_labels,
       IMB_test_features, IMB_test_labels, metric_list):
    print("--------------------------------------------")
    print("Question 4.1: Linear SVM with imbalanced class weights")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 10, 1: 1})
    clf.fit(X_train, Y_train)
    for metric in metric_list:
        if metric == "accuracy":
            Y_pred = clf.predict(X_test)
        else:
            Y_pred = clf.decision_function(X_test)
        test_perf = performance(Y_test, Y_pred, metric)
        print("Test Performance on metric " + metric + ": %.4f" % test_perf)

    ##################################################################
    print("Question 4.2: Linear SVM on an imbalanced data set")

    print("class_weight={-1: 1, 1: 1}")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 1, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    for metric in metric_list:
        if metric == "accuracy":
            Y_pred = clf.predict(IMB_test_features)
        else:
            Y_pred = clf.decision_function(IMB_test_features)
        test_perf = performance(IMB_test_labels, Y_pred, metric)
        print("Test Performance on metric " + metric + ": %.4f" % test_perf)

    ##################################################################
    print("Question 4.3(a): Choosing appropriate class weights")
    print("Wn\t\tWp\t\tMetric\t\tCV Score")
    cv_metric = 'f1_score'
    ratios = set()
    best_cv_score = 0
    best_wn = 0
    best_wp = 0
    for wn in range(5, 1, -1):
        for wp in range(1, 9):
            ratio = float(wn) / float(wp)
            if wn < wp and ratio not in ratios:
                ratios.add(ratio)
                clf = SVC(kernel='linear', C=0.01, class_weight={-1: wn, 1: wp})
                mean_cv_score = cv_performance(clf, IMB_features, IMB_labels, 5, cv_metric)
                print(wn, "\t\t", wp, "\t\t", cv_metric, "\t\t", mean_cv_score)
                if mean_cv_score > best_cv_score:
                    best_cv_score = mean_cv_score
                    best_wn = wn
                    best_wp = wp
    print("Best (Wn, Wp): ({}, {}) -- CV Score: {}".format(best_wn, best_wp, best_cv_score))

    ##################################################################
    print("Question 4.3(b): Choosing appropriate class weights")
    print("class_weight={-1: " + str(best_wn) + ", 1: " + str(best_wp) + "}")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: best_wn, 1: best_wp})
    clf.fit(IMB_features, IMB_labels)
    for metric in metric_list:
        if metric == "accuracy":
            Y_pred = clf.predict(IMB_test_features)
        else:
            Y_pred = clf.decision_function(IMB_test_features)
        test_perf = performance(IMB_test_labels, Y_pred, metric)
        print("Test Performance on metric " + metric + ": %.4f" % test_perf)

    ##################################################################
    print("Question 4.4: The ROC curve")

    # Original class weights
    print("class_weight={-1: 1, 1: 1}")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: 1, 1: 1})
    clf.fit(IMB_features, IMB_labels)
    Y_pred = clf.decision_function(IMB_test_features)

    fpr_orig, tpr_orig, _ = metrics.roc_curve(IMB_test_labels, Y_pred)
    auroc = metrics.auc(fpr_orig, tpr_orig)
    print('AUROC with W_n = 1, W_p = 1:', metrics.roc_auc_score(IMB_test_labels, Y_pred))

    # New class weights
    print("class_weight={-1: " + str(best_wn) + ", 1: " + str(best_wp) + "}")
    clf = select_classifier(penalty='l2', c=0.01, class_weight={-1: best_wn, 1: best_wp})
    clf.fit(IMB_features, IMB_labels)
    Y_pred = clf.decision_function(IMB_test_features)

    fpr_new, tpr_new, _ = metrics.roc_curve(IMB_test_labels, Y_pred)
    auroc = metrics.auc(fpr_new, tpr_new)
    print('AUROC with W_n = {}, W_p = {}:'.format(best_wn, best_wp), metrics.roc_auc_score(IMB_test_labels, Y_pred))

    # Plot
    plt.figure()
    lw = 2
    plt.plot(fpr_orig, tpr_orig, color='darkorange',
             lw=lw, label='W_n = 1, W_p = 1 (area = %0.2f)' % auroc)
    plt.plot(fpr_new, tpr_new, color='blue',
             lw=lw, label='W_n = ' + str(best_wn) + ', W_p = ' + str(best_wp) + ' curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('4.4(a) - Receiver operating characteristic with varying class weights')
    plt.legend(loc="lower right")
    plt.show()


def q5(multiclass_features, multiclass_labels, heldout_features):
    # TODO: Apply a classifier to heldout features, and then use
    # generate_challenge_labels to print the predicted labels

    print("--------------------------------------------")
    print("Question 5: Multiclass classification using library function")
    print("Using Quadratic SVM and random hyperparameter tuning")

    a = np.random.uniform(-3.2, 5.2, [20, 1])
    C_range = 10.0 ** a
    a = np.random.uniform(-3.2, 2.2, [20, 1])
    coeff_range = 10.0 ** a
    param_range = np.concatenate((C_range, coeff_range), axis=1)
    bestR, bestC = select_param_quadratic(multiclass_features, multiclass_labels, 5, "accuracy", param_range)
    print("Best C: %.6f, bestR: %.6f" % (bestC, bestR))

    clf = select_classifier(c=bestC, degree=2, r=bestR)
    clf.fit(multiclass_features, multiclass_labels)

    predicted_heldout_labels = clf.predict(heldout_features)
    generate_challenge_labels(predicted_heldout_labels, "uniqname")

    heldout_labels = load_data('data/heldoutAnswers.csv')['label'].values.copy()
    matr = metrics.confusion_matrix(heldout_labels, predicted_heldout_labels, [-1,0,1])
    print(matr)


def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
    IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
    IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

    # TODO: Questions 2, 3, 4
    q2c(X_train)
    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    q3(X_train, Y_train, X_test, Y_test, metric_list, dictionary_binary)
    q4(X_train, Y_train, X_test, Y_test, IMB_features, IMB_labels,
      IMB_test_features, IMB_test_labels, metric_list)

    # Read multiclass data
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
    heldout_features = get_heldout_reviews(multiclass_dictionary)
    q5(multiclass_features, multiclass_labels, heldout_features)


if __name__ == '__main__':
    main()

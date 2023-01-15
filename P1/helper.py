# EECS 445 - Winter 2018
# Project 1 - helper.py

import pandas as pd
import numpy as np

import project1


def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)


def get_split_binary_data():
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Also returns the dictionary used to create the feature matrices.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    X_train = pd.concat([positiveDF[:500], negativeDF[:500]]).reset_index(drop=True).copy()
    dictionary = project1.extract_dictionary(X_train)
    X_test = pd.concat([positiveDF[500:700], negativeDF[500:700]]).reset_index(drop=True).copy()
    Y_train = X_train['label'].values.copy()
    Y_test = X_test['label'].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)


def get_imbalanced_data(dictionary):
    """
    Reads in the data from data/imbalanced.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are binary as follows
        -1: poor/average
        1: good
    Input:
        dictionary: the dictionary created via get_split_binary_data
    """
    fname = "data/imbalanced.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    dataframe = pd.concat([positiveDF[:300], negativeDF[:700]]).reset_index(drop=True).copy()
    X_train = project1.generate_feature_matrix(dataframe, dictionary)
    Y_train = dataframe['label'].values.copy()

    return (X_train, Y_train)


def get_imbalanced_test(dictionary):
    """
    Reads in the data from data/dataset.csv and returns a subset of it
    reflecting an imbalanced test dataset
        -1: poor/average
        1: good
    Input:
        dictionary: the dictionary created via get_split_binary_data
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    X_test = pd.concat([positiveDF[:400], negativeDF[:100]]).reset_index(drop=True).copy()
    Y_test = X_test['label'].values.copy()
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_test, Y_test)


def get_multiclass_training_data():
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dictionary = project1.extract_dictionary(dataframe)
    X_train = project1.generate_feature_matrix(dataframe, dictionary)
    Y_train = dataframe['label'].values.copy()

    return (X_train, Y_train, dictionary)


def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = project1.generate_feature_matrix(dataframe, dictionary)
    return X


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname+'.csv', header=['label'], index=False)
    return

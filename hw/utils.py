"""
Utilities for working with pandas

Nathan Danielsen
"""

import csv
import os
import pickle


from time import strftime

import pandas as pd


def make_dummies(df, columns=None):
    'makes dummy variables for selected columns that have a single value'
    if columns != type(str):
        for col in columns:
            col_name = df[col].name
            dummy = pd.get_dummies(df[col], prefix=col_name)
            df = pd.concat([df, dummy], axis=1)
    else:
        col_name = df[columns].name
        dummy = pd.get_dummies(df[col], prefix=col_name)
        df = pd.concat([df, dummy], axis=1)
    return clean_columns(df)

def clean_columns(df):
    'cleans column names into a nice working format'
    df.columns = df.columns.str.lower()
    df.columns = [col.replace(" ", "_").replace(",", "_").replace(":", "_") for col in df.columns]

    return df

def kaggle_submission(df, cols=None, sdir=None, time=None):
    assert cols != None
    submission_df = df[cols]

    submission_df.columns = [col.capitalize() for col in submission_df.columns]

    if time:
        time_now = str(time)
    else:
        time_now = str(strftime("%Y-%m-%d %H:%M:%S"))

    filename = 'submission-%s.csv' % time_now
    
    submission_file = '%s/submission-%s.csv' % (sdir, time_now)

    submission_df.to_csv(submission_file, index=False)

    print 'Submission File: %s created' % filename


def submission_logger(time=None, features=None, models=None, test_error=None):
    """
    Creates a log of the models used to create a submission_file
    """
    csv_delimiter = '|'
    filename = 'log/submission_log.txt'
    csv_columns = ['time', 'features', 'models', 'test_error', 'kaggle_score' ]

    if not os.path.isfile(filename):
        with open(filename, 'w+') as f:
            csv_writer = csv.writer(f, delimiter=csv_delimiter)
            csv_writer.writerow(csv_columns)

    def ask_score():
        try:
            score = input("What was the score?>>>")
        except SyntaxError:
            score = 'N/A'
        return score
    if test_error == None:
        test_error = 'N/A'

    kaggle_score = ask_score()
    data = [str(time), features, models, test_error, kaggle_score]
    with open(filename, 'a+') as f:
        csv_writer = csv.writer(f, delimiter=csv_delimiter)
        csv_writer.writerow(data)


def model_logger(time=None, modeltype=None, pickle_name=None, params=None, eval_score=None, features=None):
    """
    Creates a log of models that have been created and pickled.
    Records their modeltype, timecreated, pickle_name, parameters, score-metrics and features.

    """
    csv_delimiter = '|'
    filename = 'log/pickledmodel_log.txt'
    csv_columns = ['time', 'modeltype', 'pickle_name', 'params', 'eval_score', 'features' ]

    if not os.path.isfile(filename):
        with open(filename, 'w+') as f:
            csv_writer = csv.writer(f, delimiter=csv_delimiter)
            csv_writer.writerow(csv_columns)

    data = [str(time), modeltype, pickle_name, params, eval_score, features]
    with open(filename, 'a+') as f:
        csv_writer = csv.writer(f, delimiter=csv_delimiter)
        csv_writer.writerow(data)

def pickle_loader(pickle_file=None):
    with open(pickle_file, "r") as f:  #Load model from file
        model = pickle.load(f)
        return model

def pickle_create(model=None, name=None, feature_cols=None):
    pickle_file = 'models/' + name + ''.join(feature_cols) + '.pickle'
    with open(pickle_file, "w+") as f:
        pickle.dump(model, f)


def nunique(df, columns=None):
    if not columns:
        raise Exception('Add Columns')
    for col in columns:
            num_unique = df[col].nunique()
            print "# unique %s : %s" % (col, num_unique)  

def unique(df, columns=None):
    if not columns:
        raise Exception('Add Columns')
    for col in columns:
            num_unique = df[col].unique()
            print "'%s' has %s unique values \n unique items: \n %s \n\n" % (col, len(num_unique), num_unique) 


def mapper(df, column):
    num_unique = df[column].unique()
    mapper = {user: num for num, user  in enumerate(num_unique) }
    return mapper            

def unique_mapper(df, columns=None):
    if not columns:
        raise Exception('Add Columns')
    for col in columns:
        map_dict = mapper(df, col)
        df[col] = df[col].map(map_dict)


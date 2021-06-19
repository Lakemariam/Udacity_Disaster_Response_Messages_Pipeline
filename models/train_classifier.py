import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download(['punkt', 'wordnet', 'stopwords'])

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, fbeta_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sqlalchemy import create_engine
import joblib

import warnings

warnings.simplefilter('ignore')

def load_data(database_filepath):
    """
    loading database
    Input: 
        database_filepath: .../data/DisasterResponseMessage.db
        X : message
        Y : drop the first 4 column
    Output:
        df: dataframe of the sql database
        X, Y, category_names   
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('df', engine)

    # assign X and Y values
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # take a look at unique values and data types of Y
    print(Y.related.unique())
    print(Y.dtypes)

    # return category_names
    category_names = list(df.columns[4:])

    return X, Y, category_names

def tokenize(text):
    # normalize text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Tokenize all the words in text 
    tokens = word_tokenize(text)
    
    # Initiate Lemmatization and Lemmatize the text
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:      
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

def build_model():
    # create a pipeline 
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Parameters and GridSearchCV for Y_train
    parameters = {
       'clf__estimator__n_estimators': [6, 8, 10],
       'clf__estimator__min_samples_split': [2, 3, 4],
    }
    clf = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)

    return clf 

def evaluate_model(model, X_test, Y_test, category_names):
    
   # Predict Y_test using X_test with a model pipeline
   Y_pred_test = model.predict(X_test)
   
   # Print classification report on test data (precision, recall, f1-score, support) for Y_test of category_names
   for i in range(len(category_names)):
       print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred_test[:, i]))
       print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred_test[:,i])))

def save_model(model, model_filepath):
    """
    save model output as a pickle file
    Input:
        model_filepath
    Output:
        classifier.pkl
    """
    # Open the pickle file to write in binary form
    # pickle.dump(model, open(model_filepath, 'wb'))
    pkl_filename = "classifier.pkl"
    with open(pkl_filename, 'wb') as file:
        joblib.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state = 0)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()

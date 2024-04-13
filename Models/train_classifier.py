# import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Retrieve data from an SQLite database.
    
    Args:
    database_filepath (str): Filepath to the SQLite database.
    
    Returns:
    tuple: A tuple containing two elements:
        - X (pandas.Series): Features extracted from the database.
        - Y (pandas.DataFrame): Target variables extracted from the database.
    """
    # Create an engine to connect to the SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Read data from the "WB_disaster_messages" table into a DataFrame
    df = pd.read_sql_table("WB_disaster_messages", con=engine)
    
    # Extract the 'message' column as features
    X = df['message']
    
    # Extract the target variables starting from the fifth column
    Y = df.iloc[:, 4:]
    
    return X, Y

def tokenize(text):
    """
    Tokenize input text.

    Parameters:
    text (str): Input text to be tokenized.

    Returns:
    list: A list of clean tokens.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Initialize WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Clean tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens

def build_model():
    """
    Builds a classifier model and tunes it using GridSearchCV.
    
    Returns:
    GridSearchCV: A classifier model tuned using GridSearchCV.
    
    Parameters:
    None
    """
    # Define the pipeline with vectorizer, transformer, and classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Tokenize and vectorize text
        ('tfidf', TfidfTransformer()),  # Apply TF-IDF transformation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier with Random Forest
    ])
        
    # Define parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        'clf__estimator__n_estimators': [50, 100],  # number of trees in the forest
        'clf__estimator__min_samples_split': [2, 5, 10]  # minimum number of samples required to split a node
    }
    # Create a grid search object with the pipeline and parameters
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)  # Verbose for logging
    
    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Test the model performance and produce classification report. 
    
    Parameters:
    model: Testing the classifier model.
    X_test: The test dataset.
    Y_test: The labels corresponding to the test data in X_test
    
    Returns:
    dict:Classification report for each column
    """
    # Predict labels using the model
    y_pred = model.predict(X_test)

    # Loop over each column and print classification report
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))
        
        
def save_model(model, model_filepath):
    """ 
    Export the final model as a pickle file.
    
    Parameters:
    model: The final trained model to be saved.
    model_filepath: The filepath where the model will be saved.
    """
    # Write the model to a binary file using pickle
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """ 
    Orchestrates the entire process: builds, trains, evaluates, and saves the model.
    """
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) == 3:
        # Extract database and model file paths from command-line arguments
        database_filepath, model_filepath = sys.argv[1:]
        
        # Load data from the database
        print('Retrieving data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        
        # Split data into training and testing sets
        print('Spliting data (Training and testing)...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # Build the model
        print('Building model...')
        model = build_model()
        
        # Train the model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # Testing the model
        print('Testing model...')
        evaluate_model(model, X_test, Y_test)

        # Save the trained model to a file
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Saved model!')
    else:
        # Prompt user to provide correct command-line arguments if not provided
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()

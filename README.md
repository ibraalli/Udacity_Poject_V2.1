# Disaster Response Pipeline Project

## Contents
* Project Overview
* Introduction
* Detailed Files Navigation
* Files
    * ETL Pipeline
    * ML Pipeline
    * Flask Web App
* Instructions
* Acknowledgements
 
### Project Overview
The project focuses on building a machine learning pipeline that automates the categorization of messages, enabling swift and accurate routing to the relevant disaster aid. By leveraging advanced data engineering techniques and real-time data processing capabilities, Aim to streamline the response efforts in disester recovery.

### Detailed Files Navigation
Application  

* Web app    
    * master.html # main page    
    * go.html # classification result    
* *run.py # file that runs app    


Dataset    

* disaster_categories.csv # data Categories  
* disaster_messages.csv # data to messages    
* process_data.py # data exploration and cleaning pipeline    
* WB_disaster_Database.db # database to save clean data - Table name WB_disaster_messages    


models   

* train_classifier.py # machine learning pipeline     
* classifier.pkl # saved model     


README.md    

### Files
There are three phases completed for this project. 

#### 1. ETL Pipeline
A Python script, `process_data.py`, writes a data cleaning pipeline that:

##### Read and Merge Data:
- Read two CSV files containing messages and categories data using pd.read_csv.
- Merge the two DataFrames based on a common ID column using the merge method

##### Clean Data:
- Split the 'categories' column into separate columns based on the delimiter ';' using str.split.
- Extract category names by removing the last two characters from each element in the first row of the split categories DataFrame.
- Rename the columns of the split categories DataFrame.
- Convert category values to binary format (0 or 1) by extracting the last character of each string and converting it to integer.
- Replace values of 2 with 1 in the 'related' column.
- Drop the original 'categories' column from the DataFrame.
- Concatenate the original DataFrame with the cleaned categories DataFrame.
- Remove duplicate rows from the DataFrame using drop_duplicates

##### Upload Data:
- Create a SQLite database engine using create_engine.
- Store the DataFrame into a table named 'WB_disaster_messages' in the SQLite database using to_sql.

##### Process Data:
- Read file paths for messages CSV, categories CSV, and SQLite database from user input.
- Call the above methods sequentially to read, clean, and upload the data.
- Print progress messages to inform the user about the ongoing operations.

A jupyter notebook `ETL Pipeline Preparation` was used to prepare the process_data.py.
 
#### 2. ML Pipeline
Importing Libraries: 
- All necessary libraries are imported, including pandas, numpy, nltk, sklearn, and sqlalchemy.

##### Loading Data: 
- The load_data function retrieves data from an SQLite database. It extracts the 'message' column as features (X) and the target variables (Y) from the database.

##### Tokenization: 
- The tokenize function tokenizes the input text by first using NLTK's word tokenizer and then lemmatizing each token to its root form.

##### Building Model: 
- The build_model function constructs a machine learning pipeline consisting of CountVectorizer, TfidfTransformer, and MultiOutputClassifier with RandomForestClassifier. It tunes hyperparameters using GridSearchCV.

##### Evaluating Model: 
- The evaluate_model function tests the trained model's performance using the test dataset and prints a classification report for each target variable.

##### Saving Model: 
- The save_model function exports the final trained model as a pickle file.

##### Main Function:The main function orchestrates the entire process:
- It checks if the correct number of command-line arguments is provided.
- It loads data from the provided database filepath.
- It splits the data into training and testing sets.
- It builds, trains, evaluates, and saves the model.
- If the correct number of arguments is not provided, it prompts the user to provide them.

A jupyter notebook `ML Pipeline Preparation` was used to prepare the train_classifier.py.

#### 3. Flask Web App
##### Importing Libraries: 
- Necessary libraries like Flask, pandas, nltk, plotly, and sklearn are imported.

##### DisasterResponseApp Class: 
- This class initializes the Flask application, loads data from an SQLite database, loads a pre-trained machine learning model, tokenizes and lemmatizes text, and sets up Flask routes for the web application.

##### load_data Method: 
- This method loads data from an SQLite database into a Pandas DataFrame.

##### load_model Method: 
- This method loads a pre-trained machine learning model using joblib.

##### tokenize Method: 
- This method tokenizes and lemmatizes text using NLTK.

##### setup_routes Method: 
- This method sets up Flask routes for the web application. The '/' and '/index' routes render the main page with data visualizations, while the '/go' route handles user queries and displays model results.

##### run Method: 
- This method runs the Flask web application.

##### - Main Block: 
It creates an instance of the DisasterResponseApp class and runs the Flask application if the script is executed directly.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To execute an ETL pipeline for data cleaning and database storing
        `python Dataset/process_data.py Dataset/disaster_messages.csv Dataset/disaster_categories.csv Dataset/WB_disaster_Database.db`
    - ML pipeline to be run in order to train the classifier and save
        `python Models/train_classifier.py Dataset/WB_disaster_Database.db Models/classifier.pkl`

2. To run your web application, enter the following command in the app's directory.
    `python run.py`

3. Go to http://0.0.0.0:1501/



This project is designed to build an Application Processing Interface (API) model to classify disaster response messages. The disaster response messages were real messages that were sent during disaster events, the data was from Figure Eight. A machine learning pipeline were created to categorize these disaster messages to an appropriate relief agency. Building API model took three distinct steps namely:
1.	Creating an ETL pipeline
2.	Creating a ML pipeline
3.	Building a Flask web app
Initially the ETL and ML pipelines were created using Jupiter Notebook and then later translated into a Python file to make it suitable for the project workspace provided by Udacity. The project workspace contains three folders (app, data, and model) and a README.md file. The app folder contains a folder named templates with two files (go.html and master.html) and a python file called run.py that used to run the Flask web app. The data folder has SQLite database file (DisasterResponseMessages.db), dataset from Figure Eight (disaster_messages.cv and disaster_categories.csv), and a python file (process_data.py) to process the data using the ETL pipeline. The models folder contains the pickle file (classifier.pkl) that was generated from the ML Pipeline Jupiter Notebook and a python file (train_classifier.py) to train and test the disaster response message data. There are two Jupiter Notebook files attached within the master file here (ETL_Pipeline_Preparation.ipnyp and ML_Pipeline_Preparation.ipnyp). 
The three basic steps to build the Disaster Response Messages web app are: 

1. ETL Pipeline
-	Load messages and categories dataset
-	Merge these two datasets
-	Clean the datasets
-	Stores it in SQLite database
To run the ETL pipeline 

2. ML Pipeline
-	Loads data from the SQLite database
-	Splits the dataset into training and test sets
-	Builds a text processing and machine learning pipeline
-	Trains and tunes a model using GridSearchCV
-	Outputs results on the test set
-	Exports the final model as a pickle file

3. Flask Web App
-	Created different visualizations using plotly 
The following three figures represents a visualization of Distribution of Message Genres, Distribution of Message Categories, and Number of messages for each label.

The steps to run a web app;
1. Run the following commands in the project's root directory to set up your database and model.
-	To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseMessages.db
-	To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponseMessages.db models/classifier.pkl
2. Run the following command in the app's directory to run your web app.
-	python run.py
3. Run the following in a new terminal 
-	env | grep WORK
4. Now take the WORKSPACEDOMAIN and WORKSPACEID to create the following web
Go to http://0.0.0.0:3001/
WORKSPACEDOMAIN = udacity-student-workspaces.com
WORKSPACEID = view6914b2f4

https:// SPACEID-3001. SPACEDOMAIN

The web app built using the steps above is the following. 
https://view6914b2f4-3001.udacity-student-workspaces.com/


# Udacity Disaster Response Messages Pipeline

# Introduction
During an event of disasters there is always a distress call for help. All the stakeholders in disaster response create a disaster response messaging tools that the public can use. The messages from the public to a designated emergency or relief agencies are important to be categorized into distinct groups. To increase the efficiency of any disaster response understanding the messages and creating categories is helpful. In doing so this machine learning project aims to build an Application Processing Interface (API) model to classify disaster response messages. The web app help minimize the burden for any relief agency to comb through the messages and allocate any relief effort effectively in time. The disaster response messages were real messages, that were sent during disaster events and the data source is from Figure Eight. A machine learning pipeline were created to categorize these disaster messages to an appropriate relief agency. 
Building API model web app took three steps:
```
     1. Creating an ETL pipeline
     2. Creating a ML pipeline
     3. Building a Flask web app
```
Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI) that used to process human language in a text and / or voice format. NLP helps the computer to understand text and voice data with machine learning and deep learning models. 
Machine Leaning (ML) is a branch of AI with the idea of systems can learn from data and make decisions with less human intervention. ML is a method of data analysis that automates analytical model building. ML pipeline is a sequential method of automation for ML models. 

## Steps to create a ML pipeline
In this project the disaster response messages are a collection of texts and Natural Language Toolkit (NLTK) used to analyze them. From the NLTK module tokenization and lemmatization were used to analyze the text data. The Normalization of the text is the first thing to do in NLP text analysis process and then Tokenization and Lemmatization follows. Tokenization serves to break up strings into words and punctuation while Lemmatization is helpful in removing unnecessary words by returning base dictionary words. 

Here, the pipeline was created using the Scikit-learn ML library for python. From Scikit-learn liberary a pipeline module was imported, and three modules were applied to build a ML pipeline: 1) Count Vectorization (Vect): convert a collection of text documents to a matrix of token counts 2) Term-frequency times inverse-document-frequency (TfidfTransformer): transform a count matrix to a normalized tfidf representation. 3) Multioutput and Multiclass (MultiOutputClassifier): implements multioutput regression and classification. 

The disaster message and category data then divided into test (0.3) and train (0.7) dataset. The train and test data then fitted into a ML pipeline. The test data fitted in the pipeline then used to test the ML model with addition of classification metrics (Accuracy, F1, Recall, and Precision scores).  The Scikit-learn liberary classification metrics module shows the ML model performs very well. But further the model was improved by adding more classifier estimators. After improving the test data from the model then the predicted data was exported to Pickle file in order to use it in Flask web app development.

## Steps to nuild a Flask web app

Initially the `ETL and ML pipelines` were created using Jupiter Notebook and then later translated into a Python (`.py`) file to make it suitable for the project workspace provided by Udacity. The project workspace contains three folders (app, data, and model) and a README.md file. The app folder contains a folder named templates with two files (`go.html` and `master.html`) and a python file called `run.py` that used to run the Flask web app. The data folder has SQLite database file (`DisasterResponseMessages.db`), dataset from Figure Eight (`disaster_messages.cv` and `disaster_categories.csv`), and a python file (`process_data.py`) to process the data using the ETL pipeline. The models folder contains the pickle file (`classifier.pkl`) that was generated from the ML Pipeline Jupiter Notebook and a python file (`train_classifier.py`) to train and test the disaster response message data. There are two Jupiter Notebook files attached within the master file here (`ETL_Pipeline_Preparation.ipnyp` and `ML_Pipeline_Preparation.ipnyp`). 
The three basic steps to build the Disaster Response Messages web app are:
```
1. ETL Pipeline
```
* Load messages and categories dataset
* Merge these two datasets
* Clean the datasets
* Stores it in SQLite database
```
2. ML Pipeline
```
* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file
```
3. Flask Web App
Flask web app platform is an API for python that lets developers build a web-application. 
The disaster response message was analyzed in EPL pipeline, ML pipeline, and then the interactive web-app deployed in Flask. 
The interactive figures were created using plotly liberary.
```
The following three figures represents a visualization of Distribution of Message Genres, Distribution of Message Categories, and Number of messages for each label.

The steps to run a web app;
```
1. Run the following commands in the project's root directory to set up your database and model.
* To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseMessages.db`
* To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseMessages.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
	`python run.py`
3. Run the following in a new terminal
	`env | grep WORK`
4. Now take the WORKSPACEDOMAIN and WORKSPACEID to create the following web
Go to `http://0.0.0.0:3001/``
```
The developed disaster response messages web app is:
`https://view6914b2f4-3001.udacity-student-workspaces.com/`


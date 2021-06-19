import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from joblib import load
from sqlalchemy import create_engine
import pickle


app = Flask(__name__)
   
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponseMessage.db')
df = pd.read_sql_table('DisasterResponseMessage', engine)

# load model
model = load("../models/classifier.pkl").set_params(n_jobs=1)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # data for first graph
    categories = df.iloc[:,4:]
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #data for second graph
    category_names = categories.columns.values
    category_bool = (categories != 0).sum().values
    
    # data for third graph
    row_total = categories.sum(axis=1)
    multilabel_counts = row_total.value_counts().sort_index()
    multi_labels, multi_label_counts = multilabel_counts.index, multilabel_counts.values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Genres Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # GRAPH 2 - category graph    
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_bool,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Category Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                }
            }
        },
    {
     'data': [
                Bar(
                    x=multi_labels,
                    y=multi_label_counts,
                )
            ],

            'layout': {
                'title': 'Number of labels for each messages',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of labels"
                    
                }
            }
    }    
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')

def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':    
    main()
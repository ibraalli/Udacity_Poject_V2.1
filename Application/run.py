import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


class DisasterResponseApp:
    """
    Class for running a Flask web application for disaster response message classification.
    """

    def __init__(self):
        """
        Initializes the DisasterResponseApp class.
        """
        self.app = Flask(__name__)
        self.load_data()
        self.load_model()
        self.setup_routes()

    def load_data(self):
        """
        Loads data from a SQLite database into a Pandas DataFrame.
        """
        engine = create_engine('sqlite:///../Dataset/WB_disaster_Database.db')
        self.df = pd.read_sql_table('WB_disaster_messages', engine)

    def load_model(self):
        """
        Loads a pre-trained machine learning model.
        """
        self.model = joblib.load("../models/classifier.pkl")

    def tokenize(self, text):
        """
        Tokenizes and lemmatizes text.

        Args:
        text (str): The input text to tokenize.

        Returns:
        list: A list of tokens.
        """
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens

    def setup_routes(self):
        """
        Sets up Flask routes for the web application.
        """
        @self.app.route('/')
        @self.app.route('/index')
        def index():
            """
            Renders the main page of the web application with data visualizations.
            """
            genre_counts = self.df.groupby('genre').count()['message']
            genre_names = list(genre_counts.index)
            df1 = self.df.drop(['id', 'message', 'original', 'genre'], axis=1)
            category_counts = df1.sum(axis=0)
            category_names = df1.columns
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
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Genre"
                        }
                    }
                },
                {
                    'data': [
                        Bar(
                            x=category_names,
                            y=category_counts
                        )
                    ],
                    'layout': {
                        'title': 'Distribution of Message Categories',
                        'yaxis': {
                            'title': "Count"
                        },
                        'xaxis': {
                            'title': "Category"
                        }
                    }
                }
            ]
            ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
            graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('master.html', ids=ids, graphJSON=graphJSON)

        @self.app.route('/go')
        def go():
            """
            Handles user query and displays model results.
            """
            query = request.args.get('query', '')
            classification_labels = self.model.predict([query])[0]
            classification_results = dict(zip(self.df.columns[4:], classification_labels))
            return render_template('go.html', query=query, classification_result=classification_results)

    def run(self):
        """
        Runs the Flask web application.
        """
        self.app.run(host='0.0.0.0', port=1501, debug=True)


if __name__ == '__main__':
    app = DisasterResponseApp()
    app.run()

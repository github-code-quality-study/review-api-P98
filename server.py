import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse, unquote
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

locations = [
    "Albuquerque, New Mexico",
    "Carlsbad, California",
    "Chula Vista, California",
    "Colorado Springs, Colorado",
    "Denver, Colorado",
    "El Cajon, California",
    "El Paso, Texas",
    "Escondido, California",
    "Fresno, California",
    "La Mesa, California",
    "Las Vegas, Nevada",
    "Los Angeles, California",
    "Oceanside, California",
    "Phoenix, Arizona",
    "Sacramento, California",
    "Salt Lake City, Utah",
    "Salt Lake City, Utah",
    "San Diego, California",
    "Tucson, Arizona",
]


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            query_string = environ.get("QUERY_STRING", "")
            query_params = parse_qs(query_string)

            location = unquote(query_params.get('location', [''])[0])
            start_date = query_params.get('start_date', [''])[0]
            end_date = query_params.get('end_date', [''])[0]
            for review in reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            response = sorted(reviews, key=lambda x: x["sentiment"]['compound'], reverse=True)
            if location:
                response = [review for review in response if review['Location'] == location]
            if start_date:
                response = [review for review in response if review['Timestamp'] >= start_date]
            if end_date:
                response = [review for review in response if review['Timestamp'] <= end_date]
            response_body = json.dumps(response, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                content_length = int(environ.get("CONTENT_LENGTH", 0))
            except (ValueError):
                content_length = 0
            post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            post_params = parse_qs(post_data)
            location = post_params.get('Location', [''])[0]
            review_body = post_params.get('ReviewBody', [''])[0]
            if not review_body:
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json")
                ])
                return [b'{"error": "Invalid data"}']
            if location not in locations:
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json")
                ])
                return [b'{"error": "Invalid location"}']
            sentiment = self.analyze_sentiment(review_body)
            response_body = {
                "ReviewId": str(uuid.uuid4()),
                "Location": location,
                "ReviewBody": review_body,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment
            }

            response_body = json.dumps(response_body, indent=2).encode("utf-8")
            start_response("201 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
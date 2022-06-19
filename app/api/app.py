
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse, abort
from utilities.sentiment_analysis_model import predict_pipeline
from utilities.hotel_tone_analyzer import get_hotel_normalized_sentiment, calculate_normalized_sentiment
from utilities.preprocessing import preprocess
from elasticsearch import Elasticsearch
import os

# Init flask app
app = Flask(__name__)
api = Api(app)

@app.before_first_request
def bootstrap():
    """
    Preprocesses the dataset and calculates the normalized sentiment of all hotels up front for faster inference
    Arguments:
    Returns:
        None
    """
    preprocess()
    calculate_normalized_sentiment()

# Request Parsers
sentiment_analysis_model_post_args = reqparse.RequestParser()
sentiment_analysis_model_post_args.add_argument("text", type=str, help="Text of the review is required", required=True)

hotel_tone_analyzer_post_args = reqparse.RequestParser()
hotel_tone_analyzer_post_args.add_argument("hotel", type=str, help="Hotel is required", required=True)

hotel_indexer_get_args = reqparse.RequestParser()
hotel_indexer_get_args.add_argument("hotel", type=str, help="Hotel is required", required=True)

class SentimentAnalysisModel(Resource):
    def post(self):
        """
        Returns the sentiment of a given sentence
        Arguments:
            text: text that we need to calculate sentiment for
        Returns:
            The sentiment of the given sentence
        """
        args = sentiment_analysis_model_post_args.parse_args()
        
        text = args.text

        if (not text):
            abort(422, message="Text of the review is empty")

        predictions = predict_pipeline([text])

        result = jsonify(predictions[0])
        
        return result

class HotelToneAnalyzer(Resource):
    def post(self):
        """
        Returns the normalized sentiment of a given hotel
        Arguments:
            hotel: hotel that we need to calculate the normalized sentiment for
        Returns:
            The normalized sentiment of the given hotel
        """
        args = hotel_tone_analyzer_post_args.parse_args()
        hotel = args.hotel

        # Get sentiments of reviews
        normalized_sentiment = get_hotel_normalized_sentiment(hotel)

        return normalized_sentiment.iloc[0].to_dict()

class HotelIndexer(Resource):
    def post(self):
        """
        Indexes all the hotels in the dataset along with the normalized sentiment of each hotel
        Arguments:
        Returns:
            Success message
        """
        # Connect to elasticsearch
        es = Elasticsearch(hosts=["http://localhost:9200"])

        # Delete index if exists
        es.indices.delete(index="normalized_sentiment")

        # Check if index exists
        if not es.indices.exists(index="normalized_sentiment"):
            # Calculate normalized sentiment
            normalized_sentiment = calculate_normalized_sentiment()
            
            # Create index
            es.indices.create(index="normalized_sentiment")

            # Insert records in index
            for index, document in normalized_sentiment.iterrows():
                try:
                    es.index(index="normalized_sentiment", id=index+1, document=document.to_dict())
                except Exception as e:
                    print(document.to_dict()["name"] + " indexing error")
        
        return {"message": "Data indexed successfully"}
    
    def get(self):
        """
        Returns the normalized sentiment of a given hotel from the elasticsearch index
        Arguments:
            hotel: hotel that we need to get the normalized sentiment of
        Returns:
            The normalized sentiment of the given hotel from the elasticsearch index
        """
        args = hotel_indexer_get_args.parse_args()
        hotel = args.hotel

        # Connect to elasticsearch
        es = Elasticsearch(hosts=["http://localhost:9200"])

        # Prepare query
        body = {
            "from":0,
            "size":1,
            "query": {
                "match_phrase": {
                    "name": hotel
                }
            }
        }

        res = es.search(index="normalized_sentiment", body=body)
        hits = res.body["hits"]["hits"]

        if not len(hits):
            abort(404, message="Hotel not found")
        return jsonify(hits[0])

# Adding resources to our API
api.add_resource(SentimentAnalysisModel, "/sentiment-analysis-model")
api.add_resource(HotelToneAnalyzer, "/hotel-tone-analyzer")
api.add_resource(HotelIndexer, "/hotel-indexer")

if __name__ == '__main__':
    # Run app
    app.run(host='0.0.0.0', debug=True, port=int(os.getenv('PORT', 5000)))

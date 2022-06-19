import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
from flask_restful import abort
import os
import json

# Load config file
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

# Init text blob and initialize with Naive Bayes analyzer
tb = Blobber(analyzer=NaiveBayesAnalyzer())

# Check if normalized sentiment for all hotels are calculated or not
try:
    normalized_sentiment = pd.read_csv(config["data"]["normalized_sentiment_data_path"], sep=",")
except Exception as e:
    normalized_sentiment = None

def sentiment_analyzer(review):
    """
    Returns the sentiment of a given review
    Arguments:
        review: review that we need to calculate sentiment for
    Returns:
        The sentiment of the given review
    """
    return tb(str(review)).sentiment

def get_hotel_normalized_sentiment(hotel):
    """
    Returns the normalized sentiment of a given hotel
    Arguments:
        hotel: hotel that we need to calculate the normalized sentiment for
    Returns:
        The normalized sentiment of the given hotel
    """
    normalized_sentiment = calculate_normalized_sentiment()

    hotel_normalized_sentiment = normalized_sentiment[normalized_sentiment["name"] == hotel]
    if not hotel_normalized_sentiment.empty:
        return hotel_normalized_sentiment
    else:
        abort(404, message="Hotel not found")

def calculate_normalized_sentiment():
    """
    Calculates the normalized sentiment of all hotels in the dataset and stores the output in a new CSV file
    Arguments:
    Returns:
        The normalized sentiment of all hotels in the dataset along with hotel details and hotel reviews
    """
    global normalized_sentiment

    # Loading Reviews
    reviews = pd.read_csv(config["data"]["processed_data_path"], sep=',')
    reviews = reviews.groupby(by=["name"])

    if normalized_sentiment is None:
        reviews_columns = ["reviews.date", "reviews.dateAdded", "reviews.doRecommend", "reviews.id", "reviews.rating", "reviews.text", "reviews.title", "reviews.userCity", "reviews.username", "reviews.userProvince"]
        hotel_columns = {"address", "categories", "city", "country", "latitude", "longitude", "name", "postalCode", "province"}

        # Calculate normalized sentiment of all hotels
        normalized_sentiment_list = []
        for name, hotel_reviews in reviews:
            print("Hotel: " + name)

            # Get hotel data
            hotel_data = hotel_reviews.iloc[0].to_dict()
            for reviews_column in reviews_columns:
                del hotel_data[reviews_column]

            # Get sentiments of reviews
            hotel_processed_reviews = []
            normalized_positive = 0
            normalized_negative = 0

            for index, review in hotel_reviews.iterrows():
                # Process the review
                review = review.to_dict()
                for hotel_column in hotel_columns:
                    del review[hotel_column]
                hotel_processed_reviews.append(review)

                sentiment = sentiment_analyzer(review["reviews.text"])
                normalized_positive += sentiment.p_pos
                normalized_negative += sentiment.p_neg

            total = normalized_positive + normalized_negative
            normalized_positive = normalized_positive / total
            normalized_negative = normalized_negative / total
            classification = "pos" if normalized_positive >= 0.5 else "neg"

            hotel_data["normalized_positive"] = normalized_positive
            hotel_data["normalized_negative"] = normalized_negative
            hotel_data["classification"] = classification
            hotel_data["reviews"] = hotel_processed_reviews

            normalized_sentiment_list.append(hotel_data)
        
        # combine the dataframes into one dataframes
        normalized_sentiment_df = pd.DataFrame(normalized_sentiment_list)
        # Save the processed file into the 'data' directory
        normalized_sentiment_df.to_csv(config["data"]["normalized_sentiment_data_path"], index=False)

        # Reload normalized sentiment
        normalized_sentiment = pd.read_csv(config["data"]["normalized_sentiment_data_path"], sep=",")
    
    return normalized_sentiment

if __name__=="__main__":
    pass
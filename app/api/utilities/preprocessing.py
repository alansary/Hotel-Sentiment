import json
import pandas as pd

# Load config file
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

def preprocess():
    """
    Preprocess the dataset and stores the preprocessed data in data directory
    Arguments:
    Returns:
        None
    """
    # Loading Reviews
    reviews = pd.read_csv(config["data"]["source_data_path"], sep=',')
    reviews = reviews[reviews["categories"] == "Hotels"]
    reviews.to_csv(config["data"]["processed_data_path"], index=False)
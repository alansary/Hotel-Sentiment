# Importing Required Packages
import re
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json

# Load config file
with open('config.json') as json_data_file:
    config = json.load(json_data_file)

# Loading Pipeline
with open(config["model"]["model_path"], 'rb') as f:
    loaded_pipe = pickle.load(f)

# Pipeline Prediction
def predict_pipeline(text):
    """
    Returns the sentiment of a given sentence
    Arguments:
        text: text that we need to calculate sentiment for
    Returns:
        The sentiment of the given sentence
    """
    return predict(loaded_pipe, text)

# Data Preparation
def emoji(review):
    """
    Returns the review with emojis replaced with their meaning
    Arguments:
        review: the review that we need to replace the emojis in
    Returns:
        The review with emojis replaced with their meaning
    """
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    review = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', review)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    review = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', review)
    # Love -- <3, :*
    review = re.sub(r'(<3|:\*)', ' positiveemoji ', review)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    review = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', review)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    review = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', review)
    # Cry -- :,(, :'(, :"(
    review = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', review)

    return review

def process_review(review):
    """
    Returns the review preprocessed
    Arguments:
        review: review that we need to preprocess
    Returns:
        The review preprocessed
    """
    review = review.lower()                                             # Lowercases the string
    review = re.sub('@[^\s]+', '', review)                              # Removes usernames
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', review)   # Remove URLs
    review = re.sub(r"\d+", " ", str(review))                           # Removes all digits
    review = re.sub('&quot;'," ", review)                               # Remove (&quot;) 
    review = emoji(review)                                              # Replaces Emojis
    review = re.sub(r"\b[a-zA-Z]\b", "", str(review))                   # Removes all single characters
    review = re.sub(r"[^\w\s]", " ", str(review))                       # Removes all punctuations
    review = re.sub(r'(.)\1+', r'\1\1', review)                         # Convert more than 2 letter repetitions to 2
    review = re.sub(r"\s+", " ", str(review))                           # Replaces double spaces with single space    
    return review

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """
    Returns list of text processed
    Arguments:
        text: review that we need to preprocess
    Returns:
        The list of text processed
    """
    processed_texts = []

    for review in text:
        # Process review
        review = process_review(review)

        preprocessed_words = []
        for word in review.split():
            # Check if the word is a stopword.
            if len(word) > 1 and word not in nltk.corpus.stopwords.words('english'):
                # Lemmatizing the word.
                word = lemmatizer.lemmatize(word)
                preprocessed_words.append(word)

        processed_texts.append(' '.join(preprocessed_words))

    return processed_texts

# Model Prediction
def predict(model, text):
    """
    Returns the sentiment of a given sentence
    Arguments:
        model: loaded model for inference
        text: text that we need to calculate sentiment for
    Returns:
        The sentiment of the given sentence
    """
    # Predict the sentiment
    preprocessed_text = [preprocess([review])[0] for review in text]
    predictions = model.predict(preprocessed_text)

    pred_to_label = {0: 'Negative', 1: 'Positive'}

    # Make a list of text with sentiment.
    data = []
    for t, pred in zip(text, predictions):
        data.append({'text': t, 'pred': pred, 'label': pred_to_label[pred]})

    return data

if __name__=="__main__":
    pass

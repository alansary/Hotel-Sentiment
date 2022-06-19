# Hotel Sentiment Analysis

## About

Hotel Sentiment Analysis model and indexer.

## Dependencies

* flask 2.0.0
* flask-restful 0.3.9
* nltk 3.7
* pandas 1.4.2
* textblob 0.15.3
* elasticsearch 8.2.3

## Project Structure
### ml-dev
Contains the development Python Notebook that contains the following sections:
- Data Loading & Preprocessing
- Data Exploration
- Feature Engineering
- Model Selection
- Model Evaluation
- Deployment Pipeline
### app
Contains Flask app and docker-compose to build docker image. The app structure is as follows:
- data: Contains source, processed and normalized sentiment data
- models: Contains pickled model(s)
- utilities: Contains utility functions for different services
- app.py: App entry point, contains the APIs for different services and the bootstrap function
- config.json: Contains constants used throughout the code
- docker-compose.yml: Used to build the docker image
- requirements.txt: Contains the required packages to be installed

## Usage - Conda Environment

### Create Conda Environment
```bash
$ conda create -n hotelsentiment python=3.9
$ conda activate hotelsentiment
$ conda install jupyter ipykernel
$ python3 -m ipykernel install --user --name=hotelsentiment
$ pip install -r requirements.txt
```

### Sample Requests
```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"text": "Very bad experience."}' 0.0.0.0:5000/sentiment-analysis-model
$ curl -X POST -H "Content-Type: application/json" -d '{"text": "Recommended hotel."}' 0.0.0.0:5000/sentiment-analysis-model
$ curl -X POST -H "Content-Type: application/json" -d '{"hotel": "Hotel Russo Palace"}' 0.0.0.0:5000/hotel-tone-analyzer
$ curl -X POST -H "Content-Type: application/json" -d '{}' 0.0.0.0:5000/hotel-indexer
$ curl -X GET -H "Content-Type: application/json" -d '{"hotel": "Hotel Russo Palace"}' 0.0.0.0:5000/hotel-indexer
$ curl -X GET -H "Content-Type: application/json" -d '{"hotel": "Hotel Olcott"}' 0.0.0.0:5000/hotel-indexer
```

## Usage - Docker Image

### Build Docker Image

```bash
$ docker compose up --build
```

### Sample Requests

```bash
$ curl -X POST -H "Content-Type: application/json" -d '{"text": "Very bad experience."}' localhost:5000/sentiment-analysis-model
$ curl -X POST -H "Content-Type: application/json" -d '{"text": "Recommended hotel."}' localhost:5000/sentiment-analysis-model
$ curl -X POST -H "Content-Type: application/json" -d '{"hotel": "Hotel Russo Palace"}' localhost:5000/hotel-tone-analyzer
$ curl -X POST -H "Content-Type: application/json" -d '{}' localhost:5000/hotel-indexer
$ curl -X GET -H "Content-Type: application/json" -d '{"hotel": "Hotel Russo Palace"}' localhost:5000/hotel-indexer
$ curl -X GET -H "Content-Type: application/json" -d '{"hotel": "Hotel Olcott"}' localhost:5000/hotel-indexer
```

## API Services

### SentimentAnalysisModel
Given text, outputs the sentiment based on a trained Logistic Regression model.

### HotelToneAnalyzer
Given hotel name, outputs the normalized sentiment of the hotel along with hotel data and reviews.

### HotelIndexer

#### POST
Indexes all data found in dataset for each hotel along with normalized sentiment. Note that each time we call this API, it will remove the index first if exists and recreates it again then indexing the data.
### GET
Given hotel name, it will fetch the correspoinding record from elasticsearch index.

### Note
We added a bootstrap function to preprocess the data and calculate normalized sentiment for all hotels in dataset up front to shorten the inference time, this works as bulk loader and can be modified to serve live stream using Kafka topics.

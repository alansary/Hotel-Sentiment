FROM python:3.8

COPY . /api
WORKDIR /api

RUN pip install -r requirements.txt
RUN python3 -c "import nltk; nltk.download('omw-1.4'); nltk.download('wordnet'); nltk.download('stopwords')"
RUN python3 -m textblob.download_corpora

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
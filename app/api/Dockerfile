FROM ubuntu:latest

# Copy requirements
COPY requirements.txt .

# Update
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt autoremove

# Install pip
RUN apt install -y python-pip
RUN apt install -y python3-pip
RUN pip3 install --upgrade pip

# Installing requirements
RUN pip install -r requirements.txt

RUN python3 -c "import nltk; nltk.download('omw-1.4'); nltk.download('wordnet'); nltk.download('stopwords')"

RUN python3 -m textblob.download_corpora

# Install Java
RUN apt install -y openjdk-8-jdk

# Install APT Transport
RUN apt install -y apt-transport-https

# Install Utilities
RUN apt install -y wget
RUN apt-get install ca-certificates
RUN apt-get install -y gnupg2

# Install Elasticsearch
RUN wget --no-check-certificate -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
RUN echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-7.x.list
RUN apt-get -y update
RUN apt-get autoclean
RUN apt -y autoremove
RUN apt install -y elasticsearch
RUN service elasticsearch start

# Configure Elasticsearch
RUN python3 -c 'import yaml;f=open("/etc/elasticsearch/elasticsearch.yml");y=yaml.safe_load(f);y["network.host"] = "localhost";ostream=open("/etc/elasticsearch/elasticsearch.yml", "w");print(yaml.dump(y, ostream, default_flow_style=False, sort_keys=False))'
RUN service elasticsearch restart

# Copy data
COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
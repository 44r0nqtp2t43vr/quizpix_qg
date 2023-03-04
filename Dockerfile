FROM tiangolo/uvicorn-gunicorn:python3.8-slim

COPY ./app /app
ENV MAX_WORKERS="1"
ENV WEB_CONCURRENCY="1"
ENV TIMEOUT="3600"
ENV GRACEFUL_TIMEOUT="3600"
RUN apt-get update && apt-get install git -y
RUN pip install psycopg2-binary
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python -m spacy download en_core_web_sm
RUN  apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
RUN tar -xvf  s2v_reddit_2015_md.tar.gz






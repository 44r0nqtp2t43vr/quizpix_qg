FROM tiangolo/uvicorn-gunicorn:python3.8-slim

COPY ./app /app
ENV MAX_WORKERS="3"
ENV WEB_CONCURRENCY="1"
ENV TIMEOUT="600"
ENV GRACEFUL_TIMEOUT="600"
RUN apt-get update && apt-get install git -y
RUN pip install psycopg2-binary
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python -m spacy download en_core_web_sm






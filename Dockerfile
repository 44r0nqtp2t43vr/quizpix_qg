FROM tiangolo/uvicorn-gunicorn:python3.8-slim

COPY ./app /app
ENV MAX_WORKERS="1"
ENV WEB_CONCURRENCY="1"
RUN pip install --no-cache-dir -r /app/requirements.txt






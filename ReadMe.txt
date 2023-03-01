Docker base image and installation details: https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker


1. Build create image and run docker locally:

docker build -t question_generation .
docker run --name quizpix -p 80:80 question_generation

2. Postman Instructions:

http://127.0.0.1/getquestion

{

"context" : "Donald Trump is an American media personality and businessman who served as the 45th president of the United States.",
"answer" : "American"

}


3. Google cloud Run: Push docker to Google container registry

Create a project on gcloud console.
Install Gcloud SDK from https://cloud.google.com/sdk/docs/quickstart
gcloud init

docker build . --tag gcr.io/question-generation-312318/question-generation:latest


https://cloud.google.com/container-registry/docs/advanced-authentication
gcloud auth configure-docker

docker push gcr.io/question-generation-312318/question-generation:latest

4. Deploy API using Google Cloud Run

gcloud init  ---> Choose re-initialize this configuration [default] with new settings --> Pick correct cloud project to use.


Parameters: https://cloud.google.com/sdk/gcloud/reference/run/deploy

gcloud run deploy --image gcr.io/question-generation-312318/question-generation:latest --cpu 2 --concurrency 1 --memory 4Gi --platform managed --min-instances 0 --timeout 1m --port 80


5. Postman Instructions:

https://question-generation-sma6nn7ufq-el.a.run.app

{

"context" : "Donald Trump is an American media personality and businessman who served as the 45th president of the United States.",
"answer" : "American"

}


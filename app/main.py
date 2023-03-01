from fastT5 import get_onnx_model,get_onnx_runtime_sessions,OnnxT5
from transformers import AutoTokenizer
from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QuestionRequest(BaseModel):
    context: str
    answer: str

class QuestionResponse(BaseModel):
    question: str

trained_model_path = './t5_squad_v1/'

pretrained_model_name = Path(trained_model_path).stem


encoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-encoder-quantized.onnx")
decoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-decoder-quantized.onnx")
init_decoder_path = os.path.join(trained_model_path,f"{pretrained_model_name}-init-decoder-quantized.onnx")

model_paths = encoder_path, decoder_path, init_decoder_path
model_sessions = get_onnx_runtime_sessions(model_paths)
model = OnnxT5(trained_model_path, model_sessions)

tokenizer = AutoTokenizer.from_pretrained(trained_model_path)


def get_question(sentence,answer,mdl,tknizer):
  text = "context: {} answer: {}".format(sentence,answer)
  print (text)
  max_len = 256
  encoding = tknizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = mdl.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=128)


  dec = [tknizer.decode(ids,skip_special_tokens=True) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question



@app.get('/')
def index():
    return {'message':'hello world'}

@app.post("/getquestion", response_model=QuestionResponse)
def getquestion(request: QuestionRequest):
	context = request.context
	answer = request.answer
	ques = get_question(context,answer,model,tokenizer)
	return QuestionResponse(question=ques)


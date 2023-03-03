import random
from flashtext import KeywordProcessor
import traceback  # for exception handling
import pke  # for multipartite rank
import string  # for enumerating punctuations
from nltk.tokenize import sent_tokenize
import nltk
# from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5
from transformers import AutoTokenizer
from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()


class QuestionRequest(BaseModel):
    text: str


class QuestionResponse(BaseModel):
    quiz: List[dict]


# TEXT SUMMARIZATION
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)
#

nltk.download('punkt')
nltk.download('brown')

# utility function for processing text after decoding


def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final

# for summarizing input text


def summarizer(text, model, tokenizer):
    num_words = len(text.split())
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text

    # en_max_len = 1024
    # ge_max_len = 300
    en_max_len = int(num_words * 2) + 64
    ge_max_len = int(en_max_len/2) + 64
    encoding = tokenizer.encode_plus(
        text, max_length=en_max_len, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=128,
                          max_length=ge_max_len)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary

# KEYWORD EXTRACTION


def get_nouns_multipartite(content):
    num_words = len(content.split())
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')

        pos = {'PROPN', 'NOUN'}
        extractor.candidate_selection(pos=pos)

        # build multipartite graph and rank keyphrases according to relevance
        extractor.candidate_weighting(
            alpha=1.1, threshold=0.75, method='average')
        # keyphrases = extractor.get_n_best(n = 15)
        keyphrases = extractor.get_n_best(n=15 + int((200 - num_words) / 13))

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


def get_imp_keywords(summarytext, keywords):
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords


# QUESTION GENERATION
# trained_model_path = './t5_squad_v1/'

# pretrained_model_name = Path(trained_model_path).stem


# encoder_path = os.path.join(
#     trained_model_path, f"{pretrained_model_name}-encoder-quantized.onnx")
# decoder_path = os.path.join(
#     trained_model_path, f"{pretrained_model_name}-decoder-quantized.onnx")
# init_decoder_path = os.path.join(
#     trained_model_path, f"{pretrained_model_name}-init-decoder-quantized.onnx")

# model_paths = encoder_path, decoder_path, init_decoder_path
# model_sessions = get_onnx_runtime_sessions(model_paths)
# question_model = OnnxT5(trained_model_path, model_sessions)

# question_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)

question_model = T5ForConditionalGeneration.from_pretrained(
    'ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)


def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(
        text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question


# def get_question(sentence, answer, mdl, tknizer):
#     text = "context: {} answer: {}".format(sentence, answer)
#     print(text)
#     max_len = 256
#     encoding = tknizer.encode_plus(
#         text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")

#     input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

#     outs = mdl.generate(input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         early_stopping=True,
#                         num_beams=5,
#                         num_return_sequences=1,
#                         no_repeat_ngram_size=2,
#                         max_length=128)

#     dec = [tknizer.decode(ids, skip_special_tokens=True) for ids in outs]

#     Question = dec[0].replace("question:", "")
#     Question = Question.strip()
#     return Question

def generate_quiz(text):
    summarized_text = summarizer(text, summary_model, summary_tokenizer)
    keywords = get_nouns_multipartite(text)
    imp_keywords = get_imp_keywords(summarized_text, keywords)
    imp_keywords_p1 = imp_keywords[:int(len(imp_keywords)/2)]
    imp_keywords_p2 = imp_keywords[int(len(imp_keywords)/2) + 1:]
    questions = []
    for answer in imp_keywords_p1:
        que = get_question(summarized_text, answer,
                           question_model, question_tokenizer)
        questions.append(que)
    generated_question_list = []

    for answer_ind in range(len(imp_keywords_p1)):
        generated_question = {}
        generated_question['type'] = 'multiple_choice'
        generated_question['question'] = get_question(
            summarized_text, imp_keywords_p1[answer_ind], question_model, question_tokenizer)
        generated_question['answer'] = imp_keywords_p1[answer_ind].capitalize()
        generated_question['choices'] = []
        random.shuffle(generated_question['choices'])
        generated_question_list.append(generated_question)
    return generated_question_list


@app.get('/')
def index():
    return {'message': 'hello world'}


@app.post("/generatequiz", response_model=QuestionResponse)
def getquestion(request: QuestionRequest):
    text = request.text
    quiz = generate_quiz(text)

    return QuestionResponse(quiz=quiz)

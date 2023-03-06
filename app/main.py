from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import os
from pathlib import Path
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
import string  # for enumerating punctuations
import pke  # for multipartite rank
import traceback  # for exception handling
from flashtext import KeywordProcessor
import random
import numpy as np
# from sense2vec import Sense2Vec
# s2v = Sense2Vec().from_disk('s2v_old')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
# from sentence_transformers import SentenceTransformer
# from similarity.normalized_levenshtein import NormalizedLevenshtein
# from collections import OrderedDict
# from sklearn.metrics.pairwise import cosine_similarity
import re
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk import tokenize


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

# DISTRACTOR RETRIEVAL
# sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
# normalized_levenshtein = NormalizedLevenshtein()

# def filter_same_sense_words(original, wordlist):
#   filtered_words = []
#   base_sense = original.split('|')[1] 
#   print(base_sense)
#   for eachword in wordlist:
#     if eachword[0].split('|')[1] == base_sense:
#       filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
#   return filtered_words

# def get_highest_similarity_score(wordlist, wrd):
#   score = []
#   for each in wordlist:
#     score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
#   return max(score)

# def sense2vec_get_words(word, s2v, topn, question):
#     output = []
#     try:
#       sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
#       most_similar = s2v.most_similar(sense, n=topn)
#       output = filter_same_sense_words(sense, most_similar)
#     except:
#       output = []

#     threshold = 0.6
#     final = [word]
#     checklist = question.split()
#     for x in output:
#       if get_highest_similarity_score(final,x) < threshold and x not in final and x not in checklist:
#         final.append(x)
    
#     return final[1:]

# def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

#     # Extract similarity within words, and between words and the document
#     word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
#     word_similarity = cosine_similarity(word_embeddings)

#     # Initialize candidates and already choose best keyword/keyphrase
#     keywords_idx = [np.argmax(word_doc_similarity)]
#     candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

#     for _ in range(top_n - 1):
#         # Extract similarities within candidates and
#         # between candidates and selected keywords/phrases
#         candidate_similarities = word_doc_similarity[candidates_idx, :]
#         target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

#         # Calculate MMR
#         mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
#         mmr_idx = candidates_idx[np.argmax(mmr)]

#         # Update keywords & candidates
#         keywords_idx.append(mmr_idx)
#         candidates_idx.remove(mmr_idx)

#     return [words[idx] for idx in keywords_idx]

nltk.download('omw-1.4')

def get_distractors_wordnet(word):
    distractors = []
    try:
      syn = wn.synsets(word,'n')[0]
      
      word = word.lower()
      orig_word = word
      if len(word.split()) > 0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()

          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      print ("Wordnet distractors not found")
    return distractors

# def get_distractors(word, origsentence, sense2vecmodel, top_n, lambdaval):
#   distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
#   return distractors
#   if len(distractors) == 0:
#     return distractors
#   distractors_new = [word.capitalize()]
#   distractors_new.extend(distractors)

#   embedding_sentence = origsentence + " " + word.capitalize()
#   keyword_embedding = sentencemodel.encode([embedding_sentence])
#   distractor_embeddings = sentencemodel.encode(distractors_new)

#   max_keywords = min(len(distractors_new), 5)
#   filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)

#   final = [word.capitalize()]
#   for wrd in filtered_keywords:
#     if wrd.lower() !=word.lower():
#       final.append(wrd.capitalize())
#   final = final[1:]
#   return final

#FITBQ Generation
def tokenize_sentences(text):
    text = text.strip().replace("\n"," ")
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

def get_fill_in_the_blanks(sentence_mapping):
    out = {}
    blank_sentences = []
    processed = []
    keys = []
    for key in sentence_mapping:
        if len(sentence_mapping[key]) > 0:
            sent = sentence_mapping[key][0]
            # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
            insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
            no_of_replacements =  len(re.findall(re.escape(key), sent, re.IGNORECASE))
            line = insensitive_sent.sub(' _________ ', sent)
            if (sentence_mapping[key][0] not in processed) and no_of_replacements < 2:
                blank_sentences.append(line)
                processed.append(sentence_mapping[key][0])
                keys.append(key)
    out["sentences"] = blank_sentences
    out["keys"] = keys
    return out

# T/FQ Generation

#initialize GPT2 tokenizer and model for generating sentences
# GPT2tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# GPT2model = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id = GPT2tokenizer.eos_token_id)
GPT2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
GPT2model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id = GPT2tokenizer.eos_token_id)

def get_tf_sentences(sentence_mapping):
    out = []
    for key in sentence_mapping:
        if len(sentence_mapping[key]) > 0:
            sent = sentence_mapping[key][0]
            if sent not in out:
                out.append(sentence_mapping[key][0])
    return out

def remove_from_string(main_string, sub_string):
    combined_sub_string = sub_string.replace(" ", "")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ", "")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])
    ind = 0
    first_word = sub_string.split()[0]
    while first_word not in main_string_list and ind < last_index - 1:
      first_word = sub_string.split()[ind + 1]
      ind = ind + 1
    if first_word in main_string_list:
      return " ".join(main_string_list[:main_string_list.index(first_word)])
    else:
      return " ".join(main_string_list[:last_index-1])

nltk.download('averaged_perceptron_tagger')

def falsify_sentence(sentence):
  sentence = sentence.rstrip('?:!.,;')
  text = word_tokenize(sentence)
  tagged_text = nltk.pos_tag(text)
  words = [word[0] for word in tagged_text]
  tags = [word[1] for word in tagged_text]
  if ('CC' in tags) and ((len(tags) - tags[::-1].index('CC') - 1) >= len(tags)/2):
    ind = len(tags) - tags[::-1].index('CC') - 1
  elif ('IN' in tags) and ((len(tags) - tags[::-1].index('IN') - 1) >= len(tags)/2):
    ind = len(tags) - tags[::-1].index('IN') - 1
  elif ('VB' in tags) and ((len(tags) - tags[::-1].index('VB') - 1) >= len(tags)/2):
    ind = len(tags) - tags[::-1].index('VB') - 1
  elif ('VBD' in tags) and ((len(tags) - tags[::-1].index('VBD') - 1) >= len(tags)/2):
    ind = len(tags) - tags[::-1].index('VBD') - 1
  elif ('NN' in tags) and ((len(tags) - tags[::-1].index('NN') - 1) >= len(tags)/2):
    ind = len(tags) - tags[::-1].index('NN') - 1
  else:
    ind = -1
  if (tags[ind - 1] == 'NNP') or (tags[ind - 2] == 'NNP'):
    ind = ind - 3
  substring = " ".join(words[ind:])
  substring = re.sub(r"-LRB- ", "(", substring)
  substring = re.sub(r" -RRB-", ")", substring)
  split_sentence = remove_from_string(sentence, substring)

  #encode split_sentence and generate sentence using words with >= 80% probability
  input_ids = GPT2tokenizer.encode(split_sentence, return_tensors='tf')

  maximum_length = len(input_ids[0]) + 16
  sample_outputs = GPT2model.generate(
    input_ids, 
    do_sample = True, 
    max_length = maximum_length, 
    top_p = 0.80,
    top_k = 30,
    repetition_penalty = 10.0,
    num_return_sequences = 1
  )
  
  #decode generated sentences
  #gen_sentences = []
  for sample_output in sample_outputs:
    decoded_sentence = GPT2tokenizer.decode(sample_output, skip_special_tokens = True)
    final_sentence = tokenize.sent_tokenize(decoded_sentence)[0]
    return final_sentence
    # if (final_sentence.rstrip('?:!.,;') not in sentence) or (index == len(sample_outputs) - 1):
    #   gen_sentences.append(final_sentence)

  # return gen_sentences[0]

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
        generated_question['choices'] = get_distractors_wordnet(imp_keywords_p1[answer_ind])[:3]
        generated_question['choices'].append(imp_keywords_p1[answer_ind].capitalize())
        while len(generated_question['choices']) < 4:
          rand_word = random.choice(keywords).capitalize()
          if rand_word in generated_question['choices']:
            continue
          generated_question['choices'].append(rand_word)
        random.shuffle(generated_question['choices'])
        generated_question_list.append(generated_question)
    
    sentences = tokenize_sentences(text)
    iden_keyword_sentence_mapping = get_sentences_for_keyword(imp_keywords_p2, sentences)
    fill_in_the_blanks = get_fill_in_the_blanks(iden_keyword_sentence_mapping)
    for ind in range(len(fill_in_the_blanks['sentences'])):
        generated_question = {}
        generated_question['type'] = 'identification'
        generated_question['question'] = fill_in_the_blanks['sentences'][ind]
        generated_question['answer'] = fill_in_the_blanks['keys'][ind]
        generated_question['choices'] = []
        generated_question_list.append(generated_question)
    
    all_other_keywords = [keyword for keyword in keywords if keyword not in imp_keywords]
    other_keyword_sentence_mapping = get_sentences_for_keyword(all_other_keywords, sentences)
    tf_sentences = get_tf_sentences(other_keyword_sentence_mapping)
    for index, sentence in enumerate(tf_sentences):
        generated_question = {}
        generated_question['type'] = 'true_or_false'
        randNum = random.randint(0, 1)
        if randNum == 0:
          generated_question['question'] = falsify_sentence(sentence)
          generated_question['answer'] = 'false'
        else:
          generated_question['question'] = sentence
          generated_question['answer'] = 'true'
        generated_question['choices'] = []
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

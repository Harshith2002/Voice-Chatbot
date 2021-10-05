from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
import sys
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
import string
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.
file=open('api/static/vit_info.txt','r',errors='ignore') # To read the text file we imported
raw=file.read().lower() #Converting .txt file into lower case
sent_tokens=nltk.sent_tokenize(raw) #List of sentences
word_tokens=nltk.word_tokenize(raw) #List of words
lemm=nltk.stem.WordNetLemmatizer() #Use WordNet Dictionary

def LemTokens(tokens):
    return [lemm.lemmatize(token) for token in tokens] #Lemmatization
remove_punct_dict=dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) #Normalization

greeting_inputs=("hello","hi","hii","hey") #ELIZA concept for greeting inputs
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return "hi,how can i help you"

def response1(user_input):
    bot_reply=''
    sent_tokens.append(user_input)
    TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english') #TF-IDF Approach
    tfidf=TfidfVec.fit_transform(sent_tokens)
    idx=cosine_similarity(tfidf[-1],tfidf).argsort()[0][-2] #Cosine Similarity
    flat=cosine_similarity(tfidf[-1],tfidf).flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        bot_reply=bot_reply+"I am sorry! Please ask anything about academics,admissions,placements,contact information of VIT-AP"
    else:
        bot_reply= bot_reply+sent_tokens[idx]
    return bot_reply

@api_view(["POST"])
def Chatapi(inputtext):
    try:
        user_input = str(inputtext.body.decode('utf-8'))
        if(user_input!='bye'):
            if(user_input=='thanks' or user_input=='thankyou'):
                return JsonResponse("You are welcome",safe=False)
            else:
                if(greeting(user_input)!=None):
                    return JsonResponse(greeting(user_input),safe=False)
                else:
                    return JsonResponse(response1(user_input),safe=False)
                    sent_tokens.remove(user_input)
        else:
            return JsonResponse("Bye!",safe=False)
    except:
        return JsonResponse("ServerSide Error contact Harshith",safe=False)

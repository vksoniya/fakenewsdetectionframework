import numpy as np
import pandas as pd
import os
import math
import operator
import re
import nltk
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt') # one time execution
#nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


# Extracting articles only with relevant keywords in the document:
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words

def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq

def pos_tagging(keywordset):
    #pos_tag = nltk.pos_tag(text.split())
    pos_tag = nltk.pos_tag(keywordset)
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf

def tf_idf_score(tf,idf):
    return tf*idf

def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_importance(sentence,dict_freq,sentences):
    sentence_score = 0
    sentence = remove_special_characters(str(sentence)) 
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = [] 
    no_of_sentences = len(sentences)
    pos_tagged_sentence = pos_tagging(sentence)
    for word in pos_tagged_sentence:
        if word.lower() not in Stopwords and word not in Stopwords and len(word)>1:
            word = word.lower()
            word = wordlemmatizer.lemmatize(word)
            sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
    return sentence_score


from transformers import pipeline

#Method 1: BART summarizer
def _generateExplanationSummarizer(_df_similar_text):
    summarizer = pipeline("summarization") #Default BART model 
    print("Bart model loaded")

    _summary_text = ""
    n = 1023
    _sub_summary_text = ""
    for article in _df_similar_text['SimilarText']:
        #print(article)
        subArticles = ""
        subArticles = [article[i:i+n] for i in range(0, len(article), n)]
        for i in range(len(subArticles)):
            _sub_text = ""
            _sub_text = summarizer(subArticles[i], max_length=100, min_length=30, do_sample=False)
            _sub_summary_text += str(_sub_text[0]['summary_text'])
        _summary_text = _summary_text + "...." + _sub_summary_text
      
    return _summary_text

def _generateExplanationTFIDF(_df_predictionset, _input_keywords):
    
    keywordset = []
    
    for keyword in _input_keywords:
        key = keyword[0].split(" ")
        print(key)
        for k in key:
            print(k)
            keywordset.append(str(k))
    
    keywordset = list(set(keywordset))
    print(keywordset)
        
    _summary_text = ""
    n = 20
    for text in _df_predictionset['Crawled Article Text']:
        tokenized_sentence = sent_tokenize(text)
        text = remove_special_characters(str(text))
        text = re.sub(r'\d+', '', text)
        tokenized_words_with_stopwords = word_tokenize(text)
        tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
        tokenized_words = [word for word in tokenized_words if len(word) > 1]
        tokenized_words = [word.lower() for word in tokenized_words]
        tokenized_words = lemmatize_words(tokenized_words)
        word_freq = freq(tokenized_words)
        
        no_of_sentences = int((n * len(tokenized_sentence))/100)
        #no_of_sentences = 2
        print("Number of sentence is:" + str(no_of_sentences))
        
        c = 1
        sentence_with_importance = {}
        
        for sent in tokenized_sentence:
            sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)    
            sentence_with_importance[c] = sentenceimp    
            c = c+1
            
        sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
        
        cnt = 0
        summary = []
        sentence_no = []
        
        for word_prob in sentence_with_importance:
            if cnt < no_of_sentences:
                sentence_no.append(word_prob[0])
                cnt = cnt+1
            else:
                break
                
        sentence_no.sort()
        cnt = 1
        for sentence in tokenized_sentence:
            if cnt in sentence_no:
                summary.append(sentence)
            cnt = cnt+1
        summary = " ".join(summary)
        
        #check if the summary contains the input keywords
        
        a = summary.rstrip(".")
        comments = a.split(".")
        
        #ksetwithnv = pos_tagging(keywordset)
        #print(ksetwithnv)
        
        isMatch = False
        for comment in comments:
            #print(comment)
            isMatch = any(string in comment.lower() for string in keywordset)
            #isMatch = all(string in comment.lower() for string in keywordset)
            if isMatch:
                _summary_text = _summary_text + comment + " "
    
        #Second check
        isMatch_fulltext = False
        isMatch_fulltext = all(string in a.lower() for string in keywordset)
        if isMatch_fulltext:
            print("full text match is possible!!!")
        else:
            print("full text match is NOT possible!!!")
        
        #if isMatch:
            #quotes.append(comment)
            #Check here and also an exception
        #    _summary_text = _summary_text + summary + " "
        #    if summary == "No summary":
        #        summary_text = "no similar summary created"
        #else:
            #_summary_text = _summary_text + " "
        bad_chars = ['\n', "\'", "LONDON (Reuters) -", "WINDSOR, England (Reuters) -", "with U REUTERS/Toby Melville/Pool", "Getty", "PHOTO", "LONDON —", "No Summary generated", "WASHINGTON —", "(Reuters) -", "FILE PHOTO:", "Photo : Viacheslav Lopatin", "( Shutterstock )", "REUTERS/", "Thursday EMBED >More News Videos", "co/wJGATOtDsR —", "EMBED >More News Videos", "INDIANAPOLIS (WISH) —" ]    
        if(len(_summary_text)>0):
            for i in bad_chars :
                _summary_text = _summary_text.replace(i, ' ')
        else:
            _summary_text = "No Summary generated"
          
    return _summary_text

def _generateExplanationT5(_df_predictionset, _input_keywords):
    _summary_text = getsummaryusingT5(_df_predictionset)
    return _summary_text

def generateJustificationText(_method_type, _df_predictionset, _input_keywords):
    
    if (_method_type == "BART"):
        _generated_justification_text = _generateExplanationSummarizer(_df_predictionset)
    if (_method_type == "TFIDF"):
        _generated_justification_text = _generateExplanationTFIDF(_df_predictionset, _input_keywords)  
    if (_method_type == "T5"):
        _generated_justification_text = _generateExplanationT5(_df_predictionset, _input_keywords)
        
    return _generated_justification_text
        

#rest of them needs to be cleaned up later
# function to remove stopwords
def _remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def _extract_word_embeddings():
    word_embeddings = {}
    f = open('Glove/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    return word_embeddings
    
    
def generateExplantionUsingSimilarity(_df_crawled_articles):
    sentences = []
    for s in _df_crawled_articles['Crawled Article Summary']:
        sentences.append(sent_tokenize(s))

    sentences = [y for x in sentences for y in x] # flatten list
    
    word_embeddings = _extract_word_embeddings()
    
    # Text Preprocessing
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    
    # remove stopwords from the sentences
    clean_sentences = [_remove_stopwords(r.split()) for r in clean_sentences]
    
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
        
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = []
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    print("total summaries are:" + str(len(ranked_sentences)))
    
    
    #removing duplicates
    res = [] 
    res = list(set(ranked_sentences)) 
    #x = [res.append(x) for x in ranked_sentences if x not in res] 
    f_summary_text = ""
    for i in range(10):
        f_summary_text = f_summary_text + " " + res[i][1]
        #print(res[i][1])
    
    return f_summary_text
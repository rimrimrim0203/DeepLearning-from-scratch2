#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import re

def preprocess(text):
    text = text.lower() # 소문자로 변환
    text = text.replace("."," .") # 공백을 포함해서 .
    words = re.split("\s+", text) # 공백을 기준으로 분류
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    # text안의 문장을 word id로 변환
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word

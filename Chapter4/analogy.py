#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

def analogy(a,b,c, word_to_id, id_to_word,word_matrix, top=5, answer=None):
    for word in (a,b,c):
        if word not in word_to_id:
            print("%s(을)를 찾을 수 없습니다"%word)
            return
    
    # a:b=c:?
    # b와 a의 차이를 c에 더해줌으로써 ?를 구함
    print("\n[analogy]"+a+":"+b+"="+c+":?")
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]],word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec-a_vec+c_vec
    
    # 크기가 1이 되도록 정규화 시켜줌
    query_vec = normalize(query_vec)
    #word_matrix = normalize(word_matrix)
    
    # 코싸인 유사도
    similarity = np.dot(word_matrix,query_vec)
    
    if answer is not None:
        print("==>"+answer+":"+str(np.dot(word_matrix[word_to_id[answer]]), query_vec))
        
    count = 0
    
    # 유사도가 높은 것 순서대로 인덱싱 정렬
    for i in (-1*similarity).argsort():
        if np.isnan(similarity[i]): # nan 값이면
            continue
        if id_to_word[i] in (a,b,c): # a,b,c 안에 있는 단어면
            continue
        print("{0}:{1}".format(id_to_word[i],similarity[i]))
        
        count += 1
        if count >= top:
            return


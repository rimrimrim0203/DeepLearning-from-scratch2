#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def convert_one_hot(corpus, vocab_size):
    """
    원핫 표현으로 변환
    
    : param corpus : 단어 ID목록(1차원 또는 2차원 넘파이 배열)
    : param vocab_size : 어휘 수
    : return : 원핫표현(2차원 or 3차원 numpy 배열)
    """
    
    N = corpus.shape[0]
    
    # 주로 target
    if corpus.ndim == 1: 
        one_hot = np.zeros((N, vocab_size), dtype = np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
    
    # 주로 context
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype = np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
                
    return one_hot


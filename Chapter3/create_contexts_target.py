#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def create_contexts_target(corpus, window_size = 1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus)-window_size):
        cs = [] # 해당 target의 context
        
        # 자기자신을 포함한 범위를 루프
        for t in range(-window_size, window_size + 1):
            if t==0: # 자기자신
                continue
            cs.append(corpus[idx+t]) 
            
        contexts.append(cs)
        
    return np.array(contexts), np.array(target)


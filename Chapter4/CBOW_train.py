#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
function_path = os.path.join("..","deep-learning-from-scratch-2-master")
chapter4_path = os.path.join(function_path, "ch04")
sys.path.append(function_path)
sys.path.append(chapter4_path)

import numpy as np
from common import config

import pickle
from common.trainer import Trainer
from common.optimizer import Adam

from cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

# 하이퍼파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)
    
# 모델 등 생성
model = CBOW(vocab_size, hidden_size , window_size, corpus)
optimizer = Adam()
trainer = Trainer(model,optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs # in_weight 꺼내오기
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {} # 필요한 파라미터 저장
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word

pkl_file = "cbow_params.pkl" # 파일명

# pkl_file 저장
with open(pkl_file, "wb") as f:
    pickle.dump(params, f,-1)


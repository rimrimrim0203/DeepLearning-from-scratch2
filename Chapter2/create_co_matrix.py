# 동시발생행렬 구하는 함수
import numpy as np
def create_co_matrix(corpus, vocab_size, window_size = 1):
    corpus_size = len(corpus) # 문장의 길이
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32) # 단어의 개수를 통해 com 행렬 형성
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx-i  
            right_idx = idx+i
            
            if left_idx >= 0 : # 왼쪽 경계 확인
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1 # 해당 글자의 왼쪽 문맥 +1
            
            if right_idx < corpus_size: # 오른쪽 경계 확인
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] +=1 # 해당 글자의 오른쪽 문맥 +1
    
    return co_matrix        
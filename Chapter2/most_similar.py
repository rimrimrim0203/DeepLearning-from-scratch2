# 유사도가 높은 단어를 구하는 함수 
import numpy as np
import sys
sys.path.append("..")
from function.cos_similarity import cos_similarity
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어를 꺼냄 
    if query not in word_to_id:
        print("%s(을)를 찾을 수 없습니다." %query)
        return
    
    print("\n[query]",query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size) # vocab에 저장되어 있는 각 단어와의 유사도를 담을 배열
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec) # 자기 자신과의 듀사도도 계산되어 있음
    
    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1*similarity).argsort(): # 유사도가 큰 것 부터 index 반환
        if id_to_word[i] == query:
            continue
        print("%s: %s" %(id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return
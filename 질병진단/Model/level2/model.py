import tensorflow as tf
import tensorflow.keras as keras
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from hanspell import spell_checker
import numpy as numpy
import re
import pandas as pd
from transformers import FunnelTokenizerFast, FunnelModel
from konlpy.tag import Okt
import pickle


class lv2_disease_diagnose:
    def __init__(self):
        '''
        __init__() : 초기화 함수
                    필요한 모델 불러오기
        '''
        DATA_PATH = 'C:/Users/82102/OneDrive/문서/Capstone_AI/AI/ai/질병진단/Model/level2/'
        with open(DATA_PATH + 'm_level2_estimator.pkl', 'rb') as b:
            self.m_lv2_model = pickle.load(b)

        with open(DATA_PATH +'m_level2_tfidf_vectorizer.pkl', 'rb') as c:
            self.m_tfidf = pickle.load(c)

        with open(DATA_PATH+'m_level2_dummies.txt', 'rb') as d:
            self.m_lv2_dummies = pickle.load(d)

        with open(DATA_PATH+'w_level2_estimator.pkl', 'rb') as b:
            self.w_lv2_model = pickle.load(b)

        with open(DATA_PATH+'w_level2_tfidf_vectorizer.pkl', 'rb') as c:
            self.w_tfidf = pickle.load(c)

        with open(DATA_PATH+'w_level2_dummies.txt', 'rb') as d:
            self.w_lv2_dummies = pickle.load(d)
      
    def input(self,sex,cheifcomplaint,onset,location):
        '''
        기초문진 및 진단 내용
        남녀 모델 분리
        '''

        self.data_dic={}
        
        #성별, 환자 주요 호소 증상, 증상 발생 시점, 증상 발생 위치

        self.data_dic['sex'] = str(sex)
        self.data_dic['cheifcomplaint'] = cheifcomplaint
        self.data_dic['onset'] = onset
        self.data_dic['location'] = location

        test_df = {
                'sex': str(sex),
                'cheifcomplaint': cheifcomplaint,
                'onset': onset,
                'location': location,
        }
        #데이터프레임 변환
        self.data_dic = pd.DataFrame([test_df])
        self.input_data = self.data_dic

        #남,녀 모델 분리
        #남자 0 여자 1
        if self.data_dic['sex'][0]=='1':
            self.data_dic['sex'][0]=='여자'
            self.lv2_model = self.w_lv2_model
            self.tfidf = self.w_tfidf
            self.lv2_dummies = self.w_lv2_dummies
        else:
            self.data_dic['sex'][0]=='남자'
            self.lv2_model = self.m_lv2_model
            self.tfidf = self.m_tfidf
            self.lv2_dummies = self.m_lv2_dummies
        
    def preprocess(self):
        test=self.input_data

        #1. 불용어 제거
        def erase_stopwords(text):
            stopwords = ['질문', '문의', '관련', '그대로', '계속', '답변', '선생님', '관련문의',
                      '한지', '자주', '좀', '쪽', '자꾸', '요즘', '몇개', '무조건', '하나요',
                      '안해', '경우', '최근', '및', '몇', '달', '일반', '전날', '저번',
                      '말', '일어나지', '며칠', '먹기', '지난번', '글', '때문', '너', '무',
                      '시', '잔', '뒤', '지속', '막', '것', '이건', '뭔가', '다시', '그',
                      '무슨', '안', '난', '기', '후', '거리', '뭘', '저', '뭐', '답젼',
                      '평생', '회복', '반', '감사', '의사', '보험', '학생', '제발', '살짝',
                      '느낌', '제', '대해','문제', '전','정도', '왜', '거', '가요',
                      '의심', '추천', '를', '지금', '무엇', '관해', '리', '세',
                      '로', '목적', '그냥', '거의', '고민', '다음', '이틀', '항상', '뭐', '때',
                      '요',  '이후', '혹시', '안녕하세요',
                      '안녕','선생','끼','일','식','첨부','말씀','이번','분','년','진단','밥',
                      '속','년','동안','코딩','바','평소','게','주','올해','월','외','소견','오후','병원',
                      '어머니','군데','여러분','전문가','건','아버지','주일','센티','동안','건가요',
                      '의견','건강','세일','결까요','학원','수업','밤','부모','적','가족','대학생',
                      '무언가','이게','무엇','포함','살','사진','제','가능','중','기재','아이',
                      '저녁','안심','걱정','씨','며칠','동네','어디','하루','동생','해외','얘',
                      '학년','사람','직장인','나이','키','몸무게','엄마','부탁','해석','혹','시가'
                      '의', '가', '이', '은', '들', '는', '잘', '걍', '과', '도', '을'
                      '를', '으로', '자', '에', '와', '하다', '다', '.', ',']
            temp_x = Okt().morphs(text, stem=True)
            temp_x = [word for word in temp_x if not word in stopwords]
            temp_x = re.findall(r'\w+', str(temp_x))
            temp_x = ' '.join(map(str, temp_x))
            
            return ' '.join(re.findall(r'\w+', str(temp_x)))

        #2. Nan값 처리
        def to_nan(x):
            if(x == '-'):
                x = ''
            elif(x == '아니오'):
                x = ''
            elif(x == '아뇨'):
                x = ''
            elif(x == '몰라요'):
                x = ''
            elif(x == '모릅니다'):
                x = ''
            elif(x == '모름'):
                x = ''
            elif(x == '아뇨'):
                x = ''
            elif(x == '없습니다'):
                x = ''
            elif(x == '없어요'):
                x = ''
            elif(x == '없음'):
                x = ''
            elif(x == '.'):
                x = ''    
            return x

        for i in range(len(test.columns)):
            test[test.columns[i]] = test.apply(lambda x : to_nan(x[test.columns[i]]) , axis = 1 )
        
        test['All'] =  test['cheifcomplaint'] + '. ' + test['onset'] + '. ' + test['location']

        test['All'] = test.apply(lambda x : erase_stopwords(x['All']) , axis = 1 )

        test_ = self.tfidf.transform(test['All'])

        return test_
        

    def run_model(self):
        '''
        모델 로드
        결과 출력 (Top 3) -> 레벨 2 (주요 증상 27가지 중 Top3) 결과 예측 (result_1, result_2, result_3)
                                    -> 사용자가 선택 -> 해당 주요 증상에 맞는 질문 제시
                                    
        output : result_1, result_2,result_3 (ex) 관절 통증,월경이상/월경통,유방통/유방덩이) (str)
        '''
        test_ = self.preprocess()

        top_k_result_lv2 = tf.math.top_k(self.lv2_model.predict_proba(test_), k=3, sorted=True)

        first = top_k_result_lv2[1][0][0]
        first = first.numpy()
        second = top_k_result_lv2[1][0][1]
        second = second.numpy()
        third = top_k_result_lv2[1][0][2]
        third = third.numpy()
        lv2_1 = self.lv2_dummies[first]
        lv2_2 = self.lv2_dummies[second]
        lv2_3 = self.lv2_dummies[third]

        self.result_1 = lv2_1
        self.result_2 = lv2_2
        self.result_3 = lv2_3

        print(self.result_1,self.result_2,self.result_3)


        

   

import pandas as pd
import joblib
import random
import numpy as np

class choldiet:
    def __init__(self):
        '''
        __init__() : 초기화 함수
                    필요한 모델,라벨 불러오기
        '''

        self.model = joblib.load('cholesterol.pkl')
        self.label = pd.read_csv('chol_foodlabel.csv')
       
    def input(self,height,weight,age,sex,practice):
        self.data_dic={}

        self.data_dic['height']= height
        self.data_dic['weight'] = weight
        self.data_dic['age'] = age
        self.data_dic['sex'] = sex

        normal_weight = int((height/100)*(height/100)*21)
        want_weight = normal_weight 

        self.data_dic['want_weight'] = want_weight
        self.data_dic['want_time'] = 30
        self.data_dic['practice'] = practice


    def rec(self):
        '''
            rec() :
                1. 목표체중(w_w), 목표기간(w_t) => 일 목표 소모 열량 (m_cal) 생성
                2. 비만도 측정, 분류
                3. 기초대사량 측정
                4. 하루 필요 열량 분석
                5. 총 열량, 탄,단,지 구하기
                6. 모델 예측
                7. 결과
        '''

        #1. 목표체중 : 표준 체중, 목표기간(w_t) => 일 목표 소모 열량 (m_cal) 생성
        if self.data_dic['weight']>self.data_dic['want_weight']:
            m_w=self.data_dic['weight']-self.data_dic['want_weight']
        else:
            m_w=self.data_dic['want_weight']-self.data_dic['weight']

        cal=m_w*7200
        m_cal=round(cal/self.data_dic['want_time'],2)

       
        #3. 기초대사량 측정

        if self.data_dic['sex']==0: #남
            BMR = 66.47 + (13.75 * self.data_dic['weight']) + (5 * self.data_dic['height']) - (6.76 * self.data_dic['age'])
        else: #여
            BMR = 655.1 + (9.56 * self.data_dic['weight']) + (1.85 * self.data_dic['height']) - (4.68 * self.data_dic['age'])
            BMR = round(BMR,1)

        #운동
        p_dcal=round(m_cal*0.5,2)

        #식사
        f_dcal=round(m_cal*0.5,2)

        #4. 하루 필요 열량
        if self.data_dic['practice']==1:
            d_cal = BMR * 1.2
        elif self.data_dic['practice']==2:
            d_cal = BMR * 1.375
        elif self.data_dic['practice']==3:
            d_cal = BMR * 1.55
        elif self.data_dic['practice']==4:
            d_cal = BMR * 1.725
        else:
            d_cal = BMR * 1.9


        #5. 총 열량, 탄,단,지 

        if self.data_dic['weight']>self.data_dic['want_weight']:
            #다이어트 시 열량
            today_cal=(d_cal - f_dcal)*0.4
        else:
            #몸무게 증량 시 열량
            today_cal=(d_cal + f_dcal)*0.4
        
        self.cal = today_cal
        self.tan = today_cal*0.5
        self.dan = today_cal*0.2
        self.ji = today_cal*0.3
        self.dan = 0
        self.na = 0
        self.cho = 0
    
        test={
            '열량': self.cal,
            '탄수화물': self.tan,
            '단백질': self.dan,
            '지방': self.ji,
            '당류': self.dan,
            '나트륨': self.na,
            '콜레스테롤': self.cho
        }

        test=pd.DataFrame([test])
        
        #6. 모델예측
        y=self.model.predict([test.iloc[0]])

        #7. 결과

        food_list = self.label['음식'][np.where(self.label['구간'] == y[0])[0]]
        cnt = random.randint(0, len(food_list)) 
        
        self.result = food_list.iloc[cnt]
        

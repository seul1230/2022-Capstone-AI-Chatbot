import pandas as pd
import joblib
import random
import numpy as np

class Diet:
    def __init__(self):
        '''
        __init__() : 초기화 함수
                    필요한 모델,라벨 불러오기
        '''

        self.model = joblib.load('다이어트.pkl')
        self.label=pd.read_csv('foodlabel.csv')
       
    def input(self,height,weight,age,sex,want_weight,want_time,practice):
        '''
            include() : 입력값 세팅
                        키 / 몸무게 / 나이 / 성별(0:남 1:여) / 목표체중 / 목표기간 / 평소 활동량 (1~5) 
        
        '''
        
        self.data_dic={}

        self.data_dic['height']= height
        self.data_dic['weight'] = weight
        self.data_dic['age'] = age
        self.data_dic['sex'] = sex
        self.data_dic['want_weight'] = want_weight
        self.data_dic['want_time'] = want_time
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

        #1. 목표체중(w_w), 목표기간(w_t) => 일 목표 소모 열량 (m_cal) 생성
        m_w=self.data_dic['weight']-self.data_dic['want_weight']
        cal=m_w*7200
        m_cal=round(cal/self.data_dic['want_time'],2)

       
        #2. 비만도 측정
        ob = round(self.data_dic['weight']/((self.data_dic['height']/100)*self.data_dic['height']/100)),1)

        #비만도 분류

        if ob < 18.5:
            ca_ob = "저체중"
        elif ob < 23:
            ca_ob = "정상체중"
        elif ob < 25:
            ca_ob = "비만 전 단계 (과체중)"
        elif ob < 30:
            ca_ob = "1단계 비만"
        elif ob < 35:
            ca_ob = "2단계 비만"
        else:
            ca_ob = "3단계 비만 (고도비만)" 

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

        '''

        _ 님이 w_t일간 m_w kg을 줄이기 위해서는
        매일 운동으로 p_dcal kcal 를 소모해야하고,
        식사는 하루 ( d_cal - f_dcal ) kcal 를 드시면 됩니다.

        '''

        #5. 총 열량, 탄,단,지 
        today_cal=(d_cal - f_dcal)*0.4
        
        self.cal = today_cal
        self.tan = today_cal*0.5
        self.dan = today_cal*0.2
        self.ji = today_cal*0.3
    
        test={
            '열량(kcal)': self.cal,
            '탄수화물(kcal)': self.tan,
            '단백질(kcal)': self.dan,
            '지방(kcal)': self.ji
        }

        test=pd.DataFrame([test])
        
        #6. 모델예측
        y=self.model.predict([test.iloc[0]])

        #7. 결과
        
        self.practice_cal = p_dcal
        self.food_cal = f_dcal
        
        food_list = self.label['음식'][np.where(self.label['구간'] == y[0])[0]]
        cnt = random.randint(0, len(food_list)) 

        self.result = food_list.iloc[cnt]

        

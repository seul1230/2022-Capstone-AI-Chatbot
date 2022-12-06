
from diseasemodel import disease_diagnose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_diseaseclass():
    '''
    self,height,weight,age,sex,cheifcomplaint,onset,location,duration
    ,course,experience,character,factor,associated,event,drug,social
    ,family,traumatic,past,feminity
    
    '''
    level2 = "소화불량/만성복통"
    height=180
    weight=75
    age=30
    sex=0
    cheifcomplaint="배가아파요"
    onset="1일 전"
    location="명치 부위"
    duration="지속"
    course="심해짐"
    experience="이전에도 3차례,통증은 이번보다 약했음"
    character="칼로 찢기는 듯한 통증, NRS 8점, 방사통: 등으로 퍼짐"
    factor="오른쪽으로 돌아 누우면 완화"
    associated="구토, 속쓰림, 어지러움, 갈증, 소변량 감소"
    event=""
    drug=""
    social="술: 1주일 6~7번, 식사는 매우 불규칙"
    family=""
    traumatic=""
    past=""
    feminity=""

    diseaseprogram=disease_diagnose()
    diseaseprogram.input(level2,height,weight,age,sex,cheifcomplaint,onset,location,duration,course,experience,character,factor,associated,event,drug,social,family,traumatic,past,feminity)
    diseaseprogram.run_model()

    

test_diseaseclass()

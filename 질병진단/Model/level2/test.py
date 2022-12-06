
from model import lv2_disease_diagnose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_lv2_diseaseclass():
    '''
    self,height,weight,age,sex,cheifcomplaint,onset,location,duration
    ,course,experience,character,factor,associated,event,drug,social
    ,family,traumatic,past,feminity
    
    '''
    sex=0
    cheifcomplaint="배가아파요"
    onset="1일 전"
    location="명치 부위"
    
    lv2_diseaseprogram=lv2_disease_diagnose()
    lv2_diseaseprogram.input(sex,cheifcomplaint,onset,location)
    lv2_diseaseprogram.run_model()

    

test_lv2_diseaseclass()

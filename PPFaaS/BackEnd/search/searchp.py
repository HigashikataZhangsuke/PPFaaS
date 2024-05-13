import time
import copy
import redis
from apscheduler.schedulers.background import BackgroundScheduler
#import pynvml
profiling_tp1 = {
    0:0,
    1: 72,
    2: 144,
    4: 288
}
#Maximum batch,需要是一个字典套字典,里面是每个batch自己又对应了多少的？
profiling_bt1 = {
    0:0,
    1: {1:24,2:44,4:70,8:72},
    2: {1:45,2: 84,4:140,8:144},
    4: {1:69,2:124,4:256,8:288}
}

profiling_tp2 = {
    0:0,
    1: 72,
    2: 144,
    4: 288
}
#Maximum batch,需要是一个字典套字典,里面是每个batch自己又对应了多少的？
profiling_bt2 = {
    0:0,
    1: {1:24,2:44,4:70,8:72},
    2: {1:45,2: 84,4:140,8:144},
    4: {1:69,2:124,4:256,8:288}
}

profiling_tp3 = {
    0:0,
    1: 72,
    2: 144,
    4: 288
}
#Maximum batch,需要是一个字典套字典,里面是每个batch自己又对应了多少的？
profiling_bt3 = {
    0:0,
    1: {1:24,2:44,4:70,8:72},
    2: {1:45,2: 84,4:140,8:144},
    4: {1:69,2:124,4:256,8:288}
}

profiling_tp4 = {
    0:0,
    1: 72,
    2: 144,
    4: 288
}
#Maximum batch,需要是一个字典套字典,里面是每个batch自己又对应了多少的？
profiling_bt4 = {
    0:0,
    1: {1:24,2:44,4:70,8:72},
    2: {1:45,2: 84,4:140,8:144},
    4: {1:69,2:124,4:256,8:288}
}

profiling_tp5 = {
    0:0,
    1: 72,
    2: 144,
    4: 288
}
#Maximum batch,需要是一个字典套字典,里面是每个batch自己又对应了多少的？
profiling_bt5 = {
    0:0,
    1: {1:24,2:44,4:70,8:72},
    2: {1:45,2: 84,4:140,8:144},
    4: {1:69,2:124,4:256,8:288}
}

profiling_tp6 = {
    0:0,
    1: 72,
    2: 144,
    4: 288
}
#Maximum batch,需要是一个字典套字典,里面是每个batch自己又对应了多少的？
profiling_bt6 = {
    0:0,
    1: {1:24,2:44,4:70,8:72},
    2: {1:45,2: 84,4:140,8:144},
    4: {1:69,2:124,4:256,8:288}
}

def searchprof(previouskey,rate,dict):
    min_key = None
    min_val = None
    for key, value in dict.items():
        if key > previouskey:
            val = value
            if val > rate and (min_val is None or val < min_val):
                min_val = val
                min_key = key
    #Rate too much, then use 8 as default
    if min_key == None:
        min_key = 8
    return min_key

def searchmaxb(dict,parallelism,rate,Usememory):
    if Usememory ==True:
        #For side models
        pass
    else:
        #For main models
        min_key = None
        min_val = None
        for key, value in dict.items():
            val = value
            if val > rate and (min_val is None or val < min_val):
                min_val = val
                min_key = key
        #Rate too high, use 8 batch as default
        if min_key == None:
            min_key = 8
        return min_key

def getrate(Requestnumber,Rlast):
    Requestnumber = float(Requestnumber.decode('utf-8'))
    RateR = (Requestnumber - Rlast) / 5
    Rlast = copy.deepcopy(Requestnumber)
    return RateR,Rlast


Rlast1 = 0
Rlast2 = 0
Rlast3 = 0
Rlast4 = 0
Rlast5 = 0
Rlast6 = 0

previousp1 = 1
previousp2 = 1
previousp3 = 1
previousp4 = 1
previousp5 = 1
previousp6 = 1

def searchpara(Reqnum1,paraqueue,Reqnum2,paraqueue2,Reqnum3,paraqueue3,Reqnum4,paraqueue4,Reqnum5,paraqueue5,Reqnum6,paraqueue6):
    global Rlast1
    global previousp1
    global Rlast2
    global previousp2
    global Rlast3
    global previousp3
    global Rlast4
    global previousp4
    global Rlast5
    global previousp5
    global Rlast6
    global previousp6
    #Define test list here.
    Main =['Bert1','Bert2']
    Side = ['Bert3','Bert4','Bert5','Bert6']
    KW=[]

    Requestnumber1 = Reqnum1.get('ReqBert')
    Requestnumber2 = Reqnum2.get('ReqBert')
    Requestnumber3 = Reqnum3.get('ReqBert')
    Requestnumber4 = Reqnum4.get('ReqBert')
    Requestnumber2 = Reqnum5.get('ReqBert')
    Requestnumber3 = Reqnum6.get('ReqBert')

    #If testing for doing Multiple main models:
    #Try best to distribute the model to different GPUs
    if Requestnumber1:
        Rate1,Rlast1 = getrate(Requestnumber1,Rlast1)

    if Requestnumber2:
        Rate2,Rlast2 = getrate(Requestnumber2,Rlast2)

    if Requestnumber3:
        Rate3,Rlast3 = getrate(Requestnumber3,Rlast3)

    if Requestnumber4:
        Rate4,Rlast4 = getrate(Requestnumber4,Rlast4)

    if Requestnumber5:
        Rate5,Rlast5 = getrate(Requestnumber5,Rlast5)

    if Requestnumber6:
        Rate6,Rlast6 = getrate(Requestnumber6,Rlast6)

    #Deal with all Main request first.

    if len(Main)>0:
        #Testing for the All main case
        for terms in Main:
            if terms == 'Bert1':
                if RateR1 > profiling_tp1[previousp1]:
                    previousp1 = searchprof(previousp1,RateR1,profiling_tp1)
                    pqm1maxb = searchmaxb(profiling_bt1[previousp1],previousp1,RateR1,False)
                else:
                        #keep the same
                    pqm1maxb = searchmaxb(profiling_bt1[previousp1], previousp1, RateR1, False)
                    for key in profiling_tp1.keys():
                        if key < previousp1:
                            if RateR1 < profiling_tp1[key]:
                                previousp1 = key
                                pqm1maxb = searchmaxb(profiling_bt1[key], previousp1, RateR1, False)
                            else:
                                pass
                        else:
                            pass
                paraqueue.set('Bertp',previousp1)
                paraqueue.set('bmax', pqm1maxb)
            elif terms == 'Bert2':
                if RateR2 > profiling_tp2[previousp2]:
                    previousp2 = searchprof(previousp2,RateR2,profiling_tp2)
                    pqm2maxb = searchmaxb(profiling_bt2[previousp2],previousp2,RateR2,False)
                else:
                        #keep the same
                    pqm2maxb = searchmaxb(profiling_bt2[previousp2], previousp2, RateR2, False)
                    for key in profiling_tp2.keys():
                        if key < previousp2:
                            if RateR2 < profiling_tp2[key]:
                                previousp2 = key
                                pqm2maxb = searchmaxb(profiling_bt2[key], previousp2, RateR2, False)
                            else:
                                pass
                        else:
                            pass
                paraqueue2.set('Bertp',previousp2)
                paraqueue2.set('bmax', pqm2maxb)

    if len(Side) > 0:
        #Actually here for Side I think it will follow basically same procedure. The only difference is :
        #For side models, they will have different parallelism here? I think it will.Do we need to set a hard bar for them? No I don't think so. No need for that.
        #Actually use rate to Fine tunning it.
        for terms in Side:
            if terms == 'Bert3':
                if RateR3 > profiling_tp3[previousp3]:
                    previousp3 = searchprof(previousp3,RateR3,profiling_tp3)
                    pqm3maxb = searchmaxb(profiling_bt3[previousp3],previousp3,RateR3,False)
                else:
                        #keep the same
                    pqm3maxb = searchmaxb(profiling_bt3[previousp3], previousp3, RateR3, False)
                    for key in profiling_tp3.keys():
                        if key < previousp3:
                            if RateR3 < profiling_tp3[key]:
                                previousp3 = key
                                pqm3maxb = searchmaxb(profiling_bt3[key], previousp3, RateR3, False)
                            else:
                                pass
                        else:
                            pass
                paraqueue3.set('Bertp',previousp3)
                paraqueue3.set('bmax', pqm3maxb)
            elif terms == 'Bert4':
                if RateR4 > profiling_tp4[previousp4]:
                    previousp4 = searchprof(previousp4,RateR4,profiling_tp4)
                    pqm4maxb = searchmaxb(profiling_bt4[previousp4],previousp4,RateR4,False)
                else:
                        #keep the same
                    pqm4maxb = searchmaxb(profiling_bt4[previousp4], previousp4, RateR4, False)
                    for key in profiling_tp4.keys():
                        if key < previousp4:
                            if RateR4 < profiling_tp4[key]:
                                previousp4 = key
                                pqm4maxb = searchmaxb(profiling_bt4[key], previousp4, RateR4, False)
                            else:
                                pass
                        else:
                            pass
                paraqueue4.set('Bertp',previousp4)
                paraqueue4.set('bmax', pqm4maxb)
            elif terms == 'Bert5':
                if RateR5 > profiling_tp5[previousp5]:
                    previousp5 = searchprof(previousp5,RateR5,profiling_tp5)
                    pqm5maxb = searchmaxb(profiling_bt5[previousp5],previousp5,RateR5,False)
                else:
                        #keep the same
                    pqm5maxb = searchmaxb(profiling_bt5[previousp5], previousp5, RateR5, False)
                    for key in profiling_tp5.keys():
                        if key < previousp5:
                            if RateR5 < profiling_tp5[key]:
                                previousp5 = key
                                pqm5maxb = searchmaxb(profiling_bt5[key], previousp5, RateR5, False)
                            else:
                                pass
                        else:
                            pass
                paraqueue5.set('Bertp',previousp5)
                paraqueue5.set('bmax', pqm5maxb)
            elif terms == 'Bert6':
                if RateR6 > profiling_tp6[previousp6]:
                    previousp6 = searchprof(previousp6,RateR6,profiling_tp6)
                    pqm6maxb = searchmaxb(profiling_bt6[previousp6],previousp6,RateR6,False)
                else:
                        #keep the same
                    pqm6maxb = searchmaxb(profiling_bt6[previousp6], previousp6, RateR6, False)
                    for key in profiling_tp6.keys():
                        if key < previousp6:
                            if RateR6 < profiling_tp6[key]:
                                previousp6 = key
                                pqm6maxb = searchmaxb(profiling_bt6[key], previousp6, RateR6, False)
                            else:
                                pass
                        else:
                            pass
                paraqueue6.set('Bertp',previousp6)
                paraqueue6.set('bmax', pqm6maxb)



def mainsp(Flaskbatchqueue,paraqueue,Flaskbatchqueue2,paraqueue2,Flaskbatchqueue3,paraqueue3,Flaskbatchqueue4,paraqueue4,Flaskbatchqueue5,paraqueue5,Flaskbatchqueue6,paraqueue6):
    scheduler = BackgroundScheduler()
    scheduler.add_job(searchpara, 'interval', seconds=5, args=(Flaskbatchqueue,paraqueue,Flaskbatchqueue2,paraqueue2,Flaskbatchqueue3,paraqueue3,Flaskbatchqueue4,paraqueue4,Flaskbatchqueue5,paraqueue5,Flaskbatchqueue6,paraqueue6))
    #scheduler.add_job(publishdata, 'interval', seconds=1, args=(req_pool,Redis,op_batchsize,p_value))
    scheduler.start()
    while True:
        pass
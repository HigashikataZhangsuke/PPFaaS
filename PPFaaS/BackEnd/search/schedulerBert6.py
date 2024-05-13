#container for scheduler. It supposed to be a sidecar. And will send data to infernece container.
#Input: The Requests received from Load balancer; Profiling datas;
#There are three thing scheduler will do: receive and write parallelism value to shared file; search for optimized batchsize at each time window;check SLO and batch condition, and the put the requests into queue of redis
import copy
import time
import redis
from flask import Flask, request, jsonify
import torch
import os
import base64
import io
import json
import requests
from multiprocessing import Process,Manager
import random
from apscheduler.schedulers.background import BackgroundScheduler

Rlast = 0
#Get the real latency and use it here. p as the first layer key. batchsize as the second layer keys.
profiling_latency = {
    1: {
        1: 0.385,
        2: 0.371,
        4: 0.475,
        8: 0.588,
    },
    2: {
        1: 0.268,
        2: 0.38,
        4: 0.49,
        8: 0.59,
    },
    4: {
        1: 0.27,
        2: 0.384,
        4: 0.494,
        8: 0.501,
    }
}

def searchbatchsize50(Flaskbatchqueue,Mybatchqueue,parallelismqueue):
    global Rlast
    Requestnumber = Flaskbatchqueue.get('ReqBert')
    #Here need to change. You will need to search batchsize with current pool's slo.
    currp = parallelismqueue.get('Bertp')
    currbmax = parallelismqueue.get('bmax')
    #Or at least use this interval's slo.
    minslo = Flaskbatchqueue.get('minslo')
    #print('start searching',flush=True)
    if Requestnumber and minslo:
        if Requestnumber!=0:
            intRequestnumber = float(Requestnumber.decode('utf-8'))
            intminslo = float(minslo.decode('utf-8'))
            deltaR = intRequestnumber - Rlast
            Rlast = copy.deepcopy(intRequestnumber)
            Rate = (deltaR*5)
            op_latency = 10
            P = 1
            bmax =8
            if currp:
                P= int(currp.decode('utf-8'))
            searchdic = profiling_latency[P]
            bestb=1
            if currbmax:
                bmax = int(currbmax.decode('utf-8'))
            searchlist = [2**i for i in range(int(bmax).bit_length())]
            for b in searchlist:
                if b == 1 and searchdic[b] < intminslo:
                     op_latency = Rate * searchdic[b]
                elif b > 1 and searchdic[b] < 0.5 * intminslo:
                    if b/Rate +  Rate *(searchdic[b])/b<= op_latency:
                        op_latency = b/Rate +  Rate *(searchdic[b])/b
                        bestb =b
            #Finally check if can meet slo, if not maximum the throughput
            if bestb ==1:
                if op_latency ==10 :
                    bestb = 8
            Mybatchqueue.set('Bert',bestb)
            #print(bestb,flush=True)
    else:
        Mybatchqueue.set('Bert', 1)

def mainBert6(Flaskbatchqueue,Mybatchqueue,parallelismqueue):
    scheduler = BackgroundScheduler()
    scheduler.add_job(searchbatchsize50, 'interval', seconds=0.2, args=(Flaskbatchqueue,Mybatchqueue,parallelismqueue))

    scheduler.start()
    while True:
        pass

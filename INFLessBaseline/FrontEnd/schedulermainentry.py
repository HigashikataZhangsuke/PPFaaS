#Flask works on here, receive data and pooling them.
#Other schedulers function are bounded together, just use the main entry to start them one by one.
import copy
import time
import redis
from flask import Flask, request, jsonify
import base64
import io
import json
from multiprocessing import Process,Manager
import logging
import pickle
import uuid
from logging.handlers import RotatingFileHandler
M1r0 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=0)
M1r2 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=2)
M2r0 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=0)
M2r2 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=2)
M3r0 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=0)
M3r2 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=2)
M4r0 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=0)
M4r2 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=2)
M5r0 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=0)
M5r2 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=2)
M6r0 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=0)
M6r2 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=2)
app = Flask(__name__)
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    'app.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
handler2 = RotatingFileHandler(
    'appsec.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
handler3 = RotatingFileHandler(
    'appthi.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
handler4 = RotatingFileHandler(
    'appfor.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
handler5 = RotatingFileHandler(
    'appfri.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
handler6 = RotatingFileHandler(
    'appsix.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
handler2.setFormatter(formatter)
handler3.setFormatter(formatter)
handler4.setFormatter(formatter)
handler5.setFormatter(formatter)
handler6.setFormatter(formatter)
handler1.addFilter(LogFileter('Model1'))
handler2.addFilter(LogFileter('Model2'))
handler3.addFilter(LogFileter('Model3'))
handler4.addFilter(LogFileter('Model4'))
handler5.addFilter(LogFileter('Model5'))
handler6.addFilter(LogFileter('Model6'))
logger.addHandler(handler1)
logger.addHandler(handler2)
logger.addHandler(handler3)
logger.addHandler(handler4)
logger.addHandler(handler5)
logger.addHandler(handler6)
@app.route('/', methods=['POST'])
def receive_event():
    data = request.get_json()
    if 'Bert1' in data:
        uniqid = str(uuid.uuid4())
        recvtime = time.time()
        M1r2.incr('ReqBert')
        orislo = data['SLO']
        slo = orislo + time.time()
        inputtext = data['Bert1']
        data_dict = {
            "slo": slo,
            "inputtext": inputtext,
            "orislo": orislo,
            "uniqid": uniqid
        }
        M1r0.zadd('Bert13', {json.dumps(data_dict): slo})
        logger.info(f"Received request for Model1 of {uniqid} at {recvtime}")
        return 'OK', 200

    elif 'Bert2' in data:
        uniqid = str(uuid.uuid4())
        recvtime = time.time()
        M2r2.incr('ReqBert')
        orislo = data['SLO']
        slo = orislo + time.time()
        inputtext = data['Bert2']
        data_dict = {
            "slo": slo,
            "inputtext": inputtext,
            "orislo": orislo,
            "uniqid": uniqid
        }
        M2r0.zadd('Bert13', {json.dumps(data_dict): slo})
        logger.info(f"Received request for Model2 of {uniqid} at {recvtime}")
        return 'OK', 200

    elif 'Bert3' in data:
        uniqid = str(uuid.uuid4())
        recvtime = time.time()
        M3r2.incr('ReqBert')
        orislo = data['SLO']
        slo = orislo + time.time()
        inputtext = data['Bert3']
        data_dict = {
            "slo": slo,
            "inputtext": inputtext,
            "orislo": orislo,
            "uniqid": uniqid
        }
        M3r0.zadd('Bert13', {json.dumps(data_dict): slo})
        logger.info(f"Received request for Model3 of {uniqid} at {recvtime}")
        return 'OK', 200
    elif 'Bert4' in data:
        uniqid = str(uuid.uuid4())
        recvtime = time.time()
        M4r2.incr('ReqBert')
        orislo = data['SLO']
        slo = orislo + time.time()
        inputtext = data['Bert4']
        data_dict = {
            "slo": slo,
            "inputtext": inputtext,
            "orislo": orislo,
            "uniqid": uniqid
        }
        M4r0.zadd('Bert13', {json.dumps(data_dict): slo})
        logger.info(f"Received request for Model4 of {uniqid} at {recvtime}")
        return 'OK', 200
    elif 'Bert5' in data:
        uniqid = str(uuid.uuid4())
        recvtime = time.time()
        M5r2.incr('ReqBert')
        orislo = data['SLO']
        slo = orislo + time.time()
        inputtext = data['Bert5']
        data_dict = {
            "slo": slo,
            "inputtext": inputtext,
            "orislo": orislo,
            "uniqid": uniqid
        }
        M5r0.zadd('Bert13', {json.dumps(data_dict): slo})
        logger.info(f"Received request for Model5 of {uniqid} at {recvtime}")
        return 'OK', 200
    elif 'Bert6' in data:
        uniqid = str(uuid.uuid4())
        recvtime = time.time()
        M6r2.incr('ReqBert')
        orislo = data['SLO']
        slo = orislo + time.time()
        inputtext = data['Bert6']
        data_dict = {
            "slo": slo,
            "inputtext": inputtext,
            "orislo": orislo,
            "uniqid": uniqid
        }
        M6r0.zadd('Bert13', {json.dumps(data_dict): slo})
        logger.info(f"Received request for Model6 of {uniqid} at {recvtime}")
        return 'OK', 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12346)

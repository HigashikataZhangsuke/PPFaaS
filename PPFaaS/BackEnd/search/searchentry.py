#Flask works on here, receive data and pooling them.
#Other schedulers function are bounded together, just use the main entry to start them one by one.
import copy
import time
import redis
import torch
import base64
import io
from multiprocessing import Process,Manager

from schedulerBert1 import mainBert1
from schedulerBert2 import mainBert2
from schedulerBert3 import mainBert3
from schedulerBert4 import mainBert4
from schedulerBert5 import mainBert5
from schedulerBert6 import mainBert6

import pickle
from apscheduler.schedulers.background import BackgroundScheduler
from searchp import mainsp

# Main entry of Inference and scale up
#The manager should appear here, showing his works
if __name__ == "__main__":

    M1r2 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=2)
    M1r3 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=3)
    M1r4 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=4)

    M2r2 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=2)
    M2r3 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=3)
    M2r4 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=4)

    M3r2 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=2)
    M3r3 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=3)
    M3r4 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=4)

    M4r2 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=2)
    M4r3 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=3)
    M4r4 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=4)

    M5r2 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=2)
    M5r3 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=3)
    M5r4 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=4)

    M6r2 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=2)
    M6r3 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=3)
    M6r4 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=4)


    p0= Process(target=mainsp, args=(M1r2,M1r4,M2r2,M2r4,M3r2,M3r4,M4r2,M4r4,M5r2,M5r4,M6r2,M6r4,))
    p1 = Process(target = mainBert1,args=(M1r2,M1r3,M1r4,))
    p2 = Process(target=mainBert2, args=(M2r2,M2r3,M2r4,))
    p3 = Process(target=mainBert3, args=(M3r2,M3r3,M3r4,))
    p4 = Process(target=mainBert4, args=(M4r2, M4r3, M4r4,))
    p5 = Process(target=mainBert5, args=(M5r2, M5r3, M5r4,))
    p6 = Process(target=mainBert6, args=(M6r2, M6r3, M6r4,))
    p0.start()
    p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    p1.join()
import torch
import torch.nn as nn
import redis
import io
#import torch.multiprocessing as mp
import multiprocessing as mp

from Bert26new import mainlogicbert26

from Bertpool26 import mainpool26

if __name__ == "__main__":

    queuenames=['Bert13']
    M5r0 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=0)
    M5r1 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=1)
    M5r2 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=2)
    M5r3 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=3)
    M5r4 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=4)
    p7 = mp.Process(target=mainpool26, args=(queuenames[0], M4r0, M4r1, M4r2, M4r3,))
    p8 = mp.Process(target=mainlogicbert26, args=(queuenames[0], M4r1, M4r4,))
    p7.start()
    p8.start()



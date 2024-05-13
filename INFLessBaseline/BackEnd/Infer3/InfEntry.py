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
    M3r0 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=0)
    M3r1 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=1)
    M3r2 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=2)
    M3r3 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=3)
    M3r4 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=4)
    p5 = mp.Process(target=mainpoo26, args=(queuenames[0], M3r0, M3r1, M3r2, M3r3,))
    p6 = mp.Process(target=mainlogicbert26, args=(queuenames[0], M3r1, M3r4,))



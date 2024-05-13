import torch
import torch.nn as nn
import redis
import io
#import torch.multiprocessing as mp
import multiprocessing as mp
#from Resnet50 import mainlogicres50
#from Resnet101 import mainlogicres101
#from Bert13new import mainlogicbert13
from Bert16new import mainlogicbert16
#from Bert13 import mainlogicbert13
#from Bertpool import mainpool
from Bertpool16 import mainpool16
# Main entry of Inference and scale up
if __name__ == "__main__":
    queuenames=['Bert13']
    #Assume 3 models are using redis now
    M6r0 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=0)
    M6r1 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=1)
    M6r2 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=2)
    M6r3 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=3)
    M6r4 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=4)

    p11 = mp.Process(target=mainpool6, args=(queuenames[0], M6r0, M6r1, M6r2, M6r3,))
    p12 = mp.Process(target=mainlogicbert6, args=(queuenames[0], M6r1, M6r4,))
    p11.start()
    p12.start()

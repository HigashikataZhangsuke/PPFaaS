import torch
import torch.nn as nn
import redis
import io
#import torch.multiprocessing as mp
import multiprocessing as mp
from Bert1 import mainlogicbert1
from Bert2 import mainlogicbert2
from Bert3 import mainlogicbert3
from Bert4 import mainlogicbert4
from Bert5 import mainlogicbert5
from Bert6 import mainlogicbert6
from Bertpool1 import mainpool1
from Bertpool2 import mainpool2
from Bertpool3 import mainpool3
from Bertpool4 import mainpool4
from Bertpool5 import mainpool5
from Bertpool6 import mainpool6

# Main entry of Inference and scale up
if __name__ == "__main__":
    queuenames=['Bert13']
    M1r0 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=0)
    M1r1 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=1)
    M1r2 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=2)
    M1r3 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=3)
    M1r4 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=4)

    M2r0 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=0)
    M2r1 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=1)
    M2r2 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=2)
    M2r3 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=3)
    M2r4 = redis.Redis(host='redis2-service.default.svc.cluster.local', port=6379, db=4)

    M3r0 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=0)
    M3r1 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=1)
    M3r2 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=2)
    M3r3 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=3)
    M3r4 = redis.Redis(host='redis3-service.default.svc.cluster.local', port=6379, db=4)

    M4r0 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=0)
    M4r1 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=1)
    M4r2 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=2)
    M4r3 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=3)
    M4r4 = redis.Redis(host='redis4-service.default.svc.cluster.local', port=6379, db=4)

    M5r0 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=0)
    M5r1 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=1)
    M5r2 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=2)
    M5r3 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=3)
    M5r4 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=4)

    M6r0 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=0)
    M6r1 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=1)
    M6r2 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=2)
    M6r3 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=3)
    M6r4 = redis.Redis(host='redis6-service.default.svc.cluster.local', port=6379, db=4)

    p1 = mp.Process(target=mainpool1, args=(queuenames[0], M1r0, M1r1, M1r2, M1r3,))
    p2 = mp.Process(target=mainlogicbert1, args=(queuenames[0], M1r1, M1r4,))
    p3 =mp.Process(target=mainpool2,args=(queuenames[0], M2r0, M2r1, M2r2, M2r3,))
    p4 = mp.Process(target=mainlogicbert2,args=(queuenames[0],M2r1,M2r4,))
    p5 = mp.Process(target=mainpool3, args=(queuenames[0], M3r0, M3r1, M3r2, M3r3,))
    p6 = mp.Process(target=mainlogicbert3, args=(queuenames[0],M3r1,M3r4,))
    p7 = mp.Process(target=mainpool4, args=(queuenames[0], M4r0, M4r1, M4r2, M4r3,))
    p8 = mp.Process(target=mainlogicbert4, args=(queuenames[0],M4r1,M4r4,))
    p9 = mp.Process(target=mainpool5, args=(queuenames[0], M5r0, M5r1, M5r2, M5r3,))
    p10 = mp.Process(target=mainlogicbert5, args=(queuenames[0],M5r1,M5r4,))
    p11 = mp.Process(target=mainpool6, args=(queuenames[0], M6r0, M6r1, M6r2, M6r3,))
    p12 = mp.Process(target=mainlogicbert6, args=(queuenames[0],M6r1,M6r4,))
    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()
    # p9.start()
    # p10.start()
    # p11.start()
    # p12.start()
import redis
import io
import multiprocessing as mp
from Bert13new import mainlogicbert13
from Bertpool import mainpool

if __name__ == "__main__":
    queuenames=['Bert13']
    M1r0 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=0)
    M1r1 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=1)
    M1r2 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=2)
    M1r3 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=3)
    M1r4 = redis.Redis(host='redis1-service.default.svc.cluster.local', port=6379, db=4)
    p1 = mp.Process(target=mainpool, args=(queuenames[0], M1r0, M1r1, M1r2, M1r3,))
    p2 = mp.Process(target=mainlogicbert13, args=(queuenames[0], M1r1, M1r4,))
    p1.start()
    p2.start()


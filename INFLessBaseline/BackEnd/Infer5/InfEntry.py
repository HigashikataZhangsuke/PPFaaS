import redis
import io
import multiprocessing as mp
from Bert13new import mainlogicbert13
from Bertpool import mainpool

if __name__ == "__main__":
    queuenames=['Bert13']
    M5r0 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=0)
    M5r1 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=1)
    M5r2 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=2)
    M5r3 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=3)
    M5r4 = redis.Redis(host='redis5-service.default.svc.cluster.local', port=6379, db=4)
    p9 = mp.Process(target=mainpool5, args=(queuenames[0], M5r0, M5r1, M5r2, M5r3,))
    p10 = mp.Process(target=mainlogicbert5, args=(queuenames[0], M5r1, M5r4,))
    p9.start()
    p10.start()


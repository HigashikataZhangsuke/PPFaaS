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
    p3 = mp.Process(target=mainpool16, args=(queuenames[0], M2r0, M2r1, M2r2, M2r3,))
    p4 = mp.Process(target=mainlogicbert16, args=(queuenames[0], M2r1, M2r4,))

    p3.start()
    p4.start()

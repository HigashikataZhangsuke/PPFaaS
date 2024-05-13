import torch
import torch.nn as nn
import redis
import io
import time
import pickle
import collections
import json
from transformers import BertConfig,BertTokenizerFast,BertModel
from apscheduler.schedulers.background import BackgroundScheduler
import redis
import heapq
import logging
import threading
from logging.handlers import RotatingFileHandler
class SLOPriorityQueue:
    def __init__(self):
        self.heap = []
        heapq.heapify(self.heap)

    def push(self, item):
        # item 是一个形如 (SLO, data) 的元组
        heapq.heappush(self.heap, item)

    def pop(self):
        # 弹出并返回具有最小 SLO 的元素
        return heapq.heappop(self.heap)

    def peek(self):
        # 查看具有最小 SLO 的元素的原始SLO，但不从队列中移除它
        return self.heap[0][2] if self.heap else None

    def is_empty(self):
        # 检查队列是否为空
        return not self.heap

    def get_top_k_elements(self, k):
        # 获取队列中前 K 个具有最小 SLO 的元素
        # 使用 heapq.nsmallest 来高效地实现这一点
        top_k = heapq.nsmallest(k, self.heap)
        return top_k

    def size(self):
        return len(self.heap)

    def pop_top_k_elements(self, k):
        # 取出并移除前 K 个元素，考虑到SLO可能重复
        top_k_elements = []
        for _ in range(min(k, len(self.heap))):
            top_k_elements.append(self.pop())  # 假设已实现pop方法

        return top_k_elements

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    'poolfir.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
# 将处理器添加到日志器
logger.addHandler(handler)

def non_blocking_zrem(redis, key, members):
    redis.zrem(key, *members)
def poolingandsend(quenm,SLOpool,Dataqueue,WriteQueue,SLOqueue,Batchqueue,tk):

    while True:
        #Read from redis
        BTS = 8
        if Dataqueue.zcard("Bert13") >=BTS:
            #OK have enough elements
            top_k_elements=Dataqueue.zrange("Bert13",0,BTS-1)
            #recvtime = time.time()
            sendlist = [(json.loads(elem.decode('utf-8'))["inputtext"]) for elem in top_k_elements]
            slos = [(json.loads(elem.decode('utf-8'))["slo"]) for elem in top_k_elements]
            orislos = [(json.loads(elem.decode('utf-8'))["orislo"]) for elem in top_k_elements]
            if orislos:
                SLOqueue.set('minslo', min(orislos))
            Dataqueue.zrem('Bert13',*top_k_elements)
            tensor_data = tk([item[0] for item in sendlist], padding=True, truncation=True, max_length=512, return_tensors="pt")
            uuids = [(json.loads(elem.decode('utf-8'))["uniqid"]) for elem in top_k_elements]
            #Need to check if all of them are violated
            serialized_tuple = pickle.dumps((BTS,tensor_data,slos,uuids))
            WriteQueue.rpush('Bert13',serialized_tuple)


def mainpool1(quenm,Dataqueue,WriteQueue,SLOqueue,Batchqueue):
    SLOpool = SLOPriorityQueue()
    tk = BertTokenizerFast.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    poolingandsend(quenm,SLOpool,Dataqueue,WriteQueue,SLOqueue,Batchqueue,tk)
    while True:
        pass
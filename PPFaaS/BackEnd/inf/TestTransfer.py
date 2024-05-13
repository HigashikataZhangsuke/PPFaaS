# container for Resnet50
# Input: read parallelism from shared file;batched input from redis queue;batched impl of inference
import torch
import logging
import pickle
import uuid
from logging.handlers import RotatingFileHandler
import time
import pickle
import collections
import torch.distributed as dist
from transformers import BertConfig,BertTokenizerFast,BertModel
from apscheduler.schedulers.background import BackgroundScheduler
import torch.multiprocessing as mp
import threading
# There supposed to be 4 classes of different running instances.
import os
#Useless, put it here just for let us know which model is Using
config = BertConfig.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
config.hidden_size = 2048
config.num_hidden_layers = 24
config.num_attention_heads = 32
config.max_position_embeddings = 512

class BertModelPart1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.embeddings
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(12)])  # 前12层
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        encoder_output = embedding_output
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        return encoder_output

class BertModelPart2(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(12, 24)])  # 后12层
        self.pooler = model.pooler
    def forward(self, encoder_output, attention_mask=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

class BertModelPart41(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.embeddings
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(6)])  # 前12层

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        encoder_output = embedding_output
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        return embedding_output

class BertModelPart42(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        #self.embeddings = model.embeddings
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(6,12)])  # 前12层

    def forward(self, encoder_output, attention_mask=None):
        # Assuming encoder_output is the output from the previous part
        extended_attention_mask = attention_mask[:, None, None, :]
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        return encoder_output


class BertModelPart43(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        #self.embeddings = model.embeddings
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(12, 18)])  # 前12层

    def forward(self, encoder_output, attention_mask=None):
        # Assuming encoder_output is the output from the previous part
        extended_attention_mask = attention_mask[:, None, None, :]
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        return encoder_output


class BertModelPart44(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(18, 24)])  # 后12层
        self.pooler = model.pooler

    def forward(self, encoder_output, attention_mask=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    'inffir.log',  # 日志文件的名称
    maxBytes=1024*1024*50,
    backupCount=30,  # 保留的旧日志文件数量
    encoding='utf-8'  # 日志文件的编码
)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
# 将处理器添加到日志器
logger.addHandler(handler)

def autoscalebert13(curr_p, latest_p,  bertmodel, pre_list):
    if curr_p != latest_p:
        modellist = []
        match curr_p:
            case 1:
                bertmodel.eval()
                modellist.append(bertmodel.to('cuda:0'))
            case 2:

                part1 = BertModelPart1(bertmodel).to('cuda:0')
                part2 = BertModelPart2(bertmodel).to('cuda:1')
                part1.eval()
                part2.eval()
                modellist.append(part1)
                modellist.append(part2)
            case 4:
                part1 = BertModelPart41(bertmodel).to('cuda:0')
                part2 = BertModelPart42(bertmodel).to('cuda:1')
                part3 = BertModelPart43(bertmodel).to('cuda:2')
                part4 = BertModelPart44(bertmodel).to('cuda:3')
                part1.eval()
                part2.eval()
                part3.eval()
                part4.eval()
                modellist.append(part1)
                modellist.append(part2)
                modellist.append(part3)
                modellist.append(part4)
        return modellist
    else:
        return pre_list

def checkprocessed(ProcessedReq):
    #global ProcessedReq
    print('For Main'+ str(ProcessedReq.value),flush=True)
    #print(str(time.time()),flush=True)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '13337'

def setup(rank,world_size):
    dist.init_process_group(backend = 'nccl',rank =rank,world_size = world_size)

def cleanup():
    dist.destroy_process_group()

#For parallelism = 2
def process21(rank,size,M,queue,informqueue):
    setup(rank,size)
    #torch.cuda.set_device(rank)
    #if rank==0:
    while True:
        st = time.time()
        if not queue.empty():
            data = queue.get()
            #batchsize =
            if data == 'None':
                informqueue.put('stop')
                time.sleep(1)
                cleanup()
                break
            else:
                inputids = data['input_ids'].to('cuda:0')
                attention_mask = data['attention_mask'].to('cuda:0')
                with torch.no_grad():
                    output = M(input_ids=inputids, attention_mask=attention_mask)
                informqueue.put('send')
                #Send metadata before truely sending data to the queue
                metadata = dist.isend(tensor= attention_mask)


                req = dist.isend(tensor = output,dst = 1)
                req.wait()
                et = time.time()
                #print('pt1 '+str((et-st)*1000),flush=True)

def process22(rank,worldsize,M,sampletensorlist,msklist,informqueue,batchqueue,ProcessedReq):
    setup(rank,worldsize)
    req = None
    while True:
        st = time.time()
        if not informqueue.empty():
            chra = informqueue.get()
            if chra == 'stop':
                cleanup()
                break
            #
            else:
                if not batchqueue.empty():
                    batch = batchqueue.get()
                    if batch ==1:
                        data1 = sampletensorlist[0]
                        msk = msklist[0]
                    elif batch == 2:
                        data1 = sampletensorlist[1]
                        msk = msklist[1]
                    elif batch == 4:
                        data1 = sampletensorlist[2]
                        msk = msklist[2]
                    elif batch == 8:
                        data1 = sampletensorlist[3]
                        msk = msklist[3]
                    elif batch == 16:
                        data1 = sampletensorlist[4]
                        msk = msklist[4]
                    else:
                        data1 = sampletensorlist[5]
                        msk = msklist[5]
                req = dist.irecv(tensor = data1, src =0)
                req.wait()
                with torch.no_grad():
                    output = M(encoder_output=data1, attention_mask=msk)
                et = time.time()
                #ProcessedReq.value += batch
                #print('pt2 '+str((et-st)*1000),flush=True)
                #print(et-st,flush=True)

if __name__ == 'main':
    #Scale to 2
    #Then test for new way of sending data.


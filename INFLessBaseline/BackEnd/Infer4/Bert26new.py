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
                modellist.append(bertmodel.to('cuda:4'))
            case 2:

                part1 = BertModelPart1(bertmodel).to('cuda:4')
                part2 = BertModelPart2(bertmodel).to('cuda:5')
                part1.eval()
                part2.eval()
                modellist.append(part1)
                modellist.append(part2)
            case 4:
                part1 = BertModelPart41(bertmodel).to('cuda:4')
                part2 = BertModelPart42(bertmodel).to('cuda:5')
                part3 = BertModelPart43(bertmodel).to('cuda:6')
                part4 = BertModelPart44(bertmodel).to('cuda:7')
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

# def checkprocessed(ProcessedReq):
#     #global ProcessedReq
#     print('For Main'+ str(ProcessedReq.value),flush=True)
#     #print(str(time.time()),flush=True)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '13340'

def setup(rank,world_size):
    dist.init_process_group(backend = 'nccl',rank =rank,world_size = world_size)

def cleanup():
    dist.destroy_process_group()

#For parallelism = 1 Just simply call inf function here
def process11(M,queue,batchqueue,ProcessedReq):
    while True:
        st = time.time()
        if not queue.empty():
            data = queue.get()
            if data == 'None':
                break
            else:
                if not batchqueue.empty():
                    batch = batchqueue.get()
                inputids = data['input_ids'].to('cuda:4')
                attention_mask = data['attention_mask'].to('cuda:4')
                with torch.no_grad():
                    output = M(input_ids=inputids, attention_mask=attention_mask)
                et = time.time()
                #print('P1AllInf '+ str((et-st)*1000),flush=True)
                #ProcessedReq.value += batch

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
                inputids = data['input_ids'].to('cuda:4')
                attention_mask = data['attention_mask'].to('cuda:4')
                with torch.no_grad():
                    output = M(input_ids=inputids, attention_mask=attention_mask)
                informqueue.put('send')
                msk_shape = torch.tensor(attention_mask.shape, dtype=torch.int64).to('cuda:4')
                dist.send(tensor=msk_shape, dst=1)
                dist.send(tensor=attention_mask, dst=1)
                outputshape = torch.tensor(output.shape, dtype=torch.int64).to('cuda:4')
                dist.send(tensor=outputshape, dst=1)
                req = dist.isend(tensor = output,dst = 1)
                req.wait()
                et = time.time()
                #print('pt1 '+str((et-st)*1000),flush=True)

def process22(rank,worldsize,M,informqueue,batchqueue,ProcessedReq):
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
                mask_shape = torch.empty(2, dtype=torch.int64).to('cuda:5')
                dist.recv(tensor=mask_shape, src=0)
                msktemp = torch.empty(tuple(mask_shape.tolist())).to('cuda:5')
                dist.recv(tensor=msktemp, src=0)
                inter_shape = torch.empty(3, dtype=torch.int64).to('cuda:5')
                dist.recv(tensor=inter_shape, src=0)
                data1 = torch.empty(tuple(inter_shape.tolist())).to('cuda:5')
                req = dist.irecv(tensor = data1, src =0)
                req.wait()
                with torch.no_grad():
                    output = M(encoder_output=data1, attention_mask=msktemp)
                et = time.time()
                #ProcessedReq.value += batch
                #print('pt2 '+str((et-st)*1000),flush=True)
                #print(et-st,flush=True)

#For parallelism = 4
def process41(rank,size,M,queue,informqueue):
    setup(rank,size)
    req = None
    while True:
        #st = time.time()
        if not queue.empty():
            data = queue.get()
            if data == 'None':
                informqueue.put('stop')
                time.sleep(1)
                cleanup()
                break
            else:
                inputids = data['input_ids'].to('cuda:4')
                attention_mask = data['attention_mask'].to('cuda:4')
                with torch.no_grad():
                    output = M(input_ids=inputids, attention_mask=attention_mask)
                informqueue.put('send')
                msk_shape = torch.tensor(attention_mask.shape, dtype=torch.int64).to('cuda:4')
                dist.send(tensor=msk_shape, dst=1)
                dist.send(tensor=attention_mask, dst=1)
                outputshape = torch.tensor(output.shape, dtype=torch.int64).to('cuda:4')
                dist.send(tensor=outputshape, dst=1)
                req = dist.isend(tensor = output,dst = 1)
                req.wait()
                #et = time.time()
                #logger.info(f"131Processed at {(et-st)*1000}")

def process42(rank,worldsize,M,informqueue,informqueue2,batchqueue):
    setup(rank,worldsize)
    req = None
    while True:
        #st = time.time()
        if not informqueue.empty():
            #st = time.time()
            chra = informqueue.get()
            if chra == 'stop':
                informqueue2.put('stop')
                cleanup()
                break
            else:
                if not batchqueue.empty():
                    batch = batchqueue.get()
                informqueue2.put('send')
                mask_shape = torch.empty(2, dtype=torch.int64).to('cuda:5')
                dist.recv(tensor=mask_shape, src=0)
                msktemp = torch.empty(tuple(mask_shape.tolist())).to('cuda:5')
                dist.recv(tensor=msktemp, src=0)
                inter_shape = torch.empty(3, dtype=torch.int64).to('cuda:5')
                dist.recv(tensor=inter_shape, src=0)
                data1 = torch.empty(tuple(inter_shape.tolist())).to('cuda:5')
                req = dist.irecv(tensor=data1, src=0)
                req.wait()
                with torch.no_grad():
                    output = M(encoder_output=data1, attention_mask=msktemp)
                #print('recvd',flush=True)

               #后面看一下这样改O不OK，不OK就改回去OK的版本也就是还是用req.wait()
                #informqueue2.put('send')
                dist.send(tensor=mask_shape, dst=2)
                dist.send(tensor=msktemp, dst=2)
                dist.send(tensor = inter_shape,dst=2)
                req = dist.isend(tensor=output, dst=2)
                req.wait()
                #Note mask could be reused, while for data you need modifacation.

                #et = time.time()
                #logger.info(f"132Processed at {(et-st)*1000}")
                #print('sendd', flush=True)
                #et = time.time()
                #print(st-et,flush=True)

def process43(rank,worldsize,M,informqueue,informqueue2,batchqueue):
    setup(rank, worldsize)
    req = None
    while True:
        st = time.time()
        if not informqueue.empty():
            # st = time.time()
            chra = informqueue.get()
            if chra == 'stop':
                informqueue2.put('stop')
                cleanup()
                break
            else:
                if not batchqueue.empty():
                    batch = batchqueue.get()
                informqueue2.put('send')
                mask_shape = torch.empty(2, dtype=torch.int64).to('cuda:6')
                dist.recv(tensor=mask_shape, src=1)
                msktemp = torch.empty(tuple(mask_shape.tolist())).to('cuda:6')
                dist.recv(tensor=msktemp, src=1)
                inter_shape = torch.empty(3, dtype=torch.int64).to('cuda:6')
                dist.recv(tensor=inter_shape, src=1)
                data1 = torch.empty(tuple(inter_shape.tolist())).to('cuda:6')
                req = dist.irecv(tensor=data1, src=1)
                req.wait()
                with torch.no_grad():
                    output = M(encoder_output=data1, attention_mask=msktemp)
                dist.send(tensor=mask_shape, dst=3)
                dist.send(tensor=msktemp, dst=3)
                dist.send(tensor=inter_shape, dst=3)
                req = dist.isend(tensor=output, dst=3)
                req.wait()  # 后面看一下这样改O不OK，不OK就改回去OK的版本也就是还是用req.wait()
                #et = time.time()
                #logger.info(f"133Processed at {(et-st)*1000}")

def process44(rank,worldsize,M,informqueue,batchqueue,ProcessedReq,uuidsqueue):
    setup(rank, worldsize)
    #global ProcessedReq
    req = None
    while True:
        #st = time.time()
        if not informqueue.empty():
            chra = informqueue.get()
            if chra == 'stop':
                cleanup()
                break
            else:
                if not batchqueue.empty():
                    batch = batchqueue.get()

                mask_shape = torch.empty(2, dtype=torch.int64).to('cuda:7')
                dist.recv(tensor=mask_shape, src=2)
                msktemp = torch.empty(tuple(mask_shape.tolist())).to('cuda:7')
                dist.recv(tensor=msktemp, src=2)
                inter_shape = torch.empty(3, dtype=torch.int64).to('cuda:7')
                dist.recv(tensor=inter_shape, src=2)
                data1 = torch.empty(tuple(inter_shape.tolist())).to('cuda:7')
                req = dist.irecv(tensor=data1, src=2)
                req.wait()
                with torch.no_grad():
                    output = M(encoder_output=data1, attention_mask=msktemp)
                et = time.time()
                uuids = uuidsqueue.get()
                logger.info(f"For this batch, the final execution timepoint and related ids are: {et} and {uuids}")
                #rocessedReq.value += batch
                #logger.info(f"134Processed at {(et-st)*1000}")
def monitor(Processes):
    for p in Processes:
        p.join()
#Change P function here
def Updateparallelism(Queuelist,ProcessList,curr_p,initmodel,ProcessedReq):
    if len(Queuelist)!=0 and len(ProcessList)!=0:
        #OK deal with it. Shutdown all process first:
        #Firstly put "None" stop signal to the dataqueue first
        Queuelist[0].put('None')
        #Then stop all processes
        # for i in range(len(ProcessList)):
        #     #Alright you will have 3 or 1 process
        #     ProcessList[i].join()
        monitor_thread = threading.Thread(target=monitor,args = (ProcessList,))
        monitor_thread.start()
        #After join() all processes are clearly shutdown. Now just init again with expected numbers of queues and processes.
        Queuelist = []
        ProcessList = []
        #Init queue first
        for i in range(8):
            Queuelist.append(mp.Queue())
        modelist = autoscalebert13(curr_p, 0, initmodel, [])
        if curr_p ==1:
           #Only init one here
            P1 = mp.Process(target=process11,args = (modelist[0],Queuelist[0],Queuelist[1],ProcessedReq,))
            ProcessList.append(P1)
            # P1.start()
        elif curr_p == 2:
            P1 = mp.Process(target=process21,args = (0,2,modelist[0],Queuelist[0],Queuelist[4],))
            P2 = mp.Process(target=process22,args = (1,2,modelist[1],Queuelist[4],Queuelist[1],ProcessedReq,))
            ProcessList.append(P1)
            ProcessList.append(P2)
            # P1.start()
            # P2.start()
        else:
            P1 = mp.Process(target=process41,args = (0,4,modelist[0],Queuelist[0],Queuelist[4],))
            P2 = mp.Process(target=process42,args = (1,4,modelist[1],Queuelist[4],Queuelist[5],Queuelist[1],))
            P3 = mp.Process(target=process43,args = (2,4,modelist[2],Queuelist[5],Queuelist[6],Queuelist[2],))
            P4 = mp.Process(target=process44,args = (3,4,modelist[3],Queuelist[6],Queuelist[3],ProcessedReq,Queuelist[7],))
            ProcessList.append(P1)
            ProcessList.append(P2)
            ProcessList.append(P3)
            ProcessList.append(P4)
    else:
        Queuelist = []
        ProcessList = []
        # Init queue first
        for i in range(8):
            Queuelist.append(mp.Queue())
        modelist = autoscalebert13(curr_p, 0, initmodel, [])
        #It's initialize problem. Just start processes.
        if curr_p == 1:
            # Only init one here
            P1 = mp.Process(target=process11,args = (modelist[0], Queuelist[0],Queuelist[1],ProcessedReq,))
            ProcessList.append(P1)
            # P1.start()
        elif curr_p == 2:
            #print('Im here', flush=True)
            P1 = mp.Process(target=process21,args = (0, 2, modelist[0], Queuelist[0], Queuelist[4], ))
            P2 = mp.Process(target=process22,args = (1, 2, modelist[1], Queuelist[4], Queuelist[1], ProcessedReq,))
            ProcessList.append(P1)
            ProcessList.append(P2)

        else:
            P1 = mp.Process(target=process41,args = (0, 4, modelist[0], Queuelist[0], Queuelist[4], ))
            P2 = mp.Process(
                target=process42,args = (1, 4, modelist[1],  Queuelist[4], Queuelist[5], Queuelist[1], ))
            P3 = mp.Process(
                target=process43,args = (2, 4, modelist[2],  Queuelist[5], Queuelist[6], Queuelist[2], ))
            P4 = mp.Process(target=process44,args = (3, 4, modelist[3], Queuelist[6], Queuelist[3],ProcessedReq,Queuelist[7],))
            ProcessList.append(P1)
            ProcessList.append(P2)
            ProcessList.append(P3)
            ProcessList.append(P4)

    return Queuelist,ProcessList

def mainlogicbert26(redisqueuename,MyRedisQueue,paraqueue):

    mp.set_start_method('spawn', force=True)
    batchindex = 1
    latest_p = 0
    pv = mp.Value('i',0)
    #Use half first for big models.
    bertmodel = BertModel.from_pretrained('/home/ubuntu/modelstorage/model10B')
    curr_p = 1
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(checkprocessed, 'interval', seconds=1,args=[pv,])
    # scheduler.start()
    Queuelist=[]
    ProcessList=[]
    Queuelist,ProcessList = Updateparallelism(Queuelist,ProcessList,curr_p,bertmodel,pv)
    for p in ProcessList:
        p.start()

    while True:
        # if batchindex%100 == 0:
        #     if paraqueue.get('Bertp'):
        #         curr_p = int(paraqueue.get('Bertp').decode('utf-8'))
        #     if curr_p!= latest_p:
        #         latest_p = curr_p
        #         Queuelist,ProcessList = Updateparallelism(Queuelist,ProcessList,curr_p,bertmodel,pv)
        #         for p in ProcessList:
        #             p.start()
        result = MyRedisQueue.blpop([redisqueuename], timeout=0.1)
        if result:
            _, inputtuple = result
            #Get real batch here
            BTS, tensor_data,slos,uuids = pickle.loads(inputtuple)
            Queuelist[7].put(uuids)
            Queuelist[0].put(tensor_data)
            Queuelist[1].put(BTS)
            Queuelist[2].put(BTS)
            Queuelist[3].put(BTS)
            batchindex+=1
            logger.info(f"For this batch, the slo and related ids are: {slos} and {uuids}")


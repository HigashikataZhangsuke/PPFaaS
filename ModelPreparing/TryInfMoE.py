from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import time
# 步骤1: 加载预训练模型的配置
config = BertConfig.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
#Bert
#1B:2048,24,32,512
#1.6B:2560,32,32,512
#2.6B: 4096,24,32,512
#3.4B: 4096,32,32,512
#7.1B：5120,48,32,512
#MoE
#Used same as Bert model
config.hidden_size = 4096
config.num_hidden_layers = 24
config.num_attention_heads = 32
config.max_position_embeddings = 512

# 步骤3: 使用修改后的配置创建一个新的BERT模型
#model = BertModel(config)
#print("Create Finished")
# 步骤4: 初始化新模型的权重（transformers库在模型创建时自动进行了权重的随机初始化）
class BertMoEModel(nn.Module):
    def __init__(self, config):
        super(BertMoEModel, self).__init__()
        # 使用配置创建基础的Bert模型
        #self.bert = BertModel(config)
        # 定义两个专家（这里简单复制Bert模型的部分层作为专家）
        self.expert1 = BertModel(config).encoder.layer[-1]
        self.expert2 = BertModel(config).encoder.layer[-2]
        # 定义门控网络，用于选择专家
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # 假设有2个专家
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs, attention_mask=None):
        #outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #encoder_output = outputs.last_hidden_state

        # 假设所有输入都将导致相同的门控决策，只需要基于一个样本（例如批量中的第一个）进行决策
        gate_output = self.gate(inputs[0][:, 0, :])  # 使用CLS token的输出作为门控输入
        gate_index = torch.argmax(gate_output[0], dim=0)  # 只需要查看批量中第一个样本的决策
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # 根据门控决策选择专家
        if gate_index == 0:
            selected_expert = self.expert1
        else:
            selected_expert = self.expert2
        #print('selected '+str(gate_index),flush=True)
        # 将整个批量通过选定的专家进行处理
        #for layer in selected_expert:
        encoder_output = selected_expert(inputs[0], attention_mask)[0]

        return encoder_output


#moe_model = BertMoEModel(config)
Model1 = BertModel.from_pretrained('C:\\Users\\14707\\Desktop\\FinalizedVersion\\Oursys\\inferencepod\\Infercontainer\\Model26B\\')
secpart = BertMoEModel(config)
secpart.load_state_dict(torch.load(('C:\\Users\\14707\\Desktop\\FinalizedVersion\\Oursys\\inferencepod\\Infercontainer\\ModelMoE10B\\moe_model.pth')))
# 步骤5: 保存模型到本地

tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
input_texts = ['Generate digital startup ideas based on the wish of the people. For example, when I say I wish there is a big large mall in my small town, you generate a business plan for the digital startup complete with idea name, a short one liner, target user persona, users pain points to solve, main value propositions, sales &amp; marketing channels, revenue stream sources, cost structures, key activities, key resources, key partners, and potential business challenges to look for. Write the result in a markdown table.']*8
#
encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
id = encoded_inputs['input_ids']
msk = encoded_inputs['attention_mask']
# output1 = Model1(input_ids=id, attention_mask=msk)
# output2 = secpart(inputs = output1,attention_mask=msk)
class BertModelPart1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.embeddings
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(9)])  # 前12层

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
        self.encoder = torch.nn.ModuleList([model.encoder.layer[i] for i in range(9, 24)])  # 后12层
        self.pooler = model.pooler

    def forward(self, encoder_output, attention_mask=None):
        extended_attention_mask = attention_mask[:, None, None, :]
        for layer in self.encoder:
            encoder_output = layer(encoder_output, extended_attention_mask)[0]
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output

output1 = BertModelPart1(Model1)(input_ids=id, attention_mask=msk)
output2 = BertModelPart2(Model1)(output1,msk)
# print("Generate input finished")
#
# device0 = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
# # 注意：确保模型部分和输入数据都在相同的设备上，这里假设使用CPU
#
# #part1 = BertModelPart1(config).to(device0)
# start = time.time()
# part2 = BertModelPart2(config).to(device1)
# stop = time.time()
# print(str(stop-start))
# # input_ids = encoded_inputs['input_ids'].to(device0)
# # attention_mask = encoded_inputs['attention_mask'].to(device0)
# # print("Move finished")
# curr=time.time()
# part2.to(device0)
# end =time.time()
# print(str(curr-end))
# 使用模型的第一部分进行推理
# with torch.no_grad():
#     part1_output = part1(input_ids=input_ids, attention_mask=attention_mask)
#     part1_output = part1_output.to(device1)
# print("Pt1 Batch inf finished")
# with torch.no_grad():
#     part2_output = part2(encoder_output=part1_output, attention_mask=attention_mask.to(device1))
# print(part2_output)
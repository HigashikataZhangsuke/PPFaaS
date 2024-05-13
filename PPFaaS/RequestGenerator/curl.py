# import threading
# import time
# import csv
# import subprocess
# import datetime
# import requests
# import json
# import uuid
# from datetime import datetime
# #import torch
# import random
# import os
# import base64
# import io
# from multiprocessing import Process
# import random
# import string
# import aiohttp
# import asyncio
# #有多少个model就define多少个send。
# broker_url = "http://broker-ingress.knative-eventing.svc.cluster.local/default/default"
#
# def generateinfdata(iters,batch):
#
#     #inputs = []
#     testtexts = ['Generate digital startup ideas based on the wish of the people. For example, when I say I wish there is a big large mall in my small town, you generate a business plan for the digital startup complete with idea name, a short one liner, target user persona, users pain points to solve, main value propositions, sales &amp; marketing channels, revenue stream sources, cost structures, key activities, key resources, key partners, and potential business challenges to look for. Write the result in a markdown table.']
#     # tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
#     #for i in range(iters):
#     #    input = []
#     #for j in range(batch):
#     #inputs.append(testtexts[5])
#         #inputs.append(input)
#     return testtexts
#
# async def fetch(session, url, json_data, headers):
#     async with session.post(url, json=json_data, headers=headers) as response:
#         return response.status
#
#
# async def continuous_request(url, json_data, headers, interval, duration):
#     async with aiohttp.ClientSession() as session:
#         start_time = time.time()
#         end_time = start_time + duration
#         tasks = []  # 用于收集所有任务
#         while time.time() < end_time:
#             #print('start send one', flush=True)
#             task = asyncio.create_task(fetch(session, url, json_data, headers))
#             tasks.append(task)
#             await asyncio.sleep(interval)  # 控制发送请求的速率
#
#         # 等待所有稳定期间的任务完成
#         responses = await asyncio.gather(*tasks, return_exceptions=True)
#
#
# async def run_async_tasks(text):
#     #print('start mainlogic', flush=True)
#     input = text
#     random_SLO = 10000
#     json_data = {
#         "Bert1": input,
#         "SLO": random_SLO
#     }
#     headers = {
#         "Ce-Id": str(uuid.uuid4()),
#         "Ce-Specversion": "1.0",
#         "Ce-Type": "your-event-type",
#         "Ce-Source": "/source/curlpod",
#         "Content-Type": "application/json"
#     }
#     #For latency, it cannot be to bursty!
#     #total_requests = 1000  # 例如，发送1000个请求
#     interval = 0.001 # 每个请求之间的间隔，根据需要调整 500 req per sec
#     duration = 180  # 测试持续时间，例如60秒
#     #warmup_duration = 5  # 前5秒为热身期
#     #cooldown_duration = 5  # 后5秒为冷却期
#     requests_sent = await continuous_request(broker_url, json_data, headers, interval, duration)
#
#
# def run_in_process(tensor):
#     print('start asyncrun', flush=True)
#     asyncio.run(run_async_tasks(tensor))
#
#
# if __name__ == '__main__':
#     processes = []
#     tensors = generateinfdata(5, 1)
#     #for _ in range(2):
#     #print('start process',flush=True)
#     p0 = Process(target=run_in_process, args=(tensors,))
#     #p1 = Process(target=run_in_process, args=(tensors,))
#     #p2 = Process(target=run_in_process, args=(tensors,))
#     #p3 = Process(target=run_in_process, args=(tensors,))
#     p0.start()
#     #p1.start()
#     #p2.start()
#     #p3.start()
#     # while True:
#     #     pass
#     processes.append(p0)
#     #processes.append(p1)
#     #processes.append(p2)
#     #processes.append(p3)
#     for p in processes:
#         p.join()


#Testing Possion
import numpy as np

def poisson_process(lamda, duration):
    """
    模拟一个泊松过程

    :param lamda: 单位时间内事件发生的平均次数（率λ）
    :param duration: 模拟的总时间长度
    :return: 一个列表，包含事件发生的时间点
    """
    events = []
    current_time = 0

    while current_time < duration:
        # 在泊松过程中，事件之间的时间间隔遵循指数分布
        # 指数分布的参数是λ（lamda）
        time_to_next_event = np.random.exponential(1 / lamda)
        current_time += time_to_next_event
        if current_time < duration:
            events.append(time_to_next_event)

    return events

# 示例：以λ=5的速率模拟一个持续时间为10单位的泊松过程
lamda = 20  # 单位时间内事件发生的平均次数
duration = 10  # 模拟的总时间长度
events = poisson_process(lamda, duration)

print("事件发生的时间点：", events)

#Testing Gaussian
# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_gamma_distribution(rate, cv, sample_size=10000):
#     # 根据Rate和CV计算Gamma分布的参数
#     alpha = (1 / cv) ** 2  # 形状参数
#     beta = 1 / (rate * alpha)  # 尺度参数
#
#     # 生成Gamma分布的样本
#     samples = np.random.gamma(alpha, beta, sample_size)
#
#     # 绘制样本的直方图
#     plt.figure(figsize=(10, 6))
#     plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')
#     plt.title(f'Gamma Distribution with Rate {rate} and CV {cv}')
#     plt.xlabel('Interval')
#     plt.ylabel('Density')
#     plt.grid(True)
#     plt.show()
#
# # 使用示例参数绘制Gamma分布
# rate = 5  # 平均速率
# cv = 0.1  # 变异系数
# plot_gamma_distribution(rate, cv)
# import numpy as np
#
# def gamma_process_intervals(rate, cv, duration, min_interval=0.001):
#     """
#     生成符合特定Gamma分布的时间间隔
#
#     :param rate: 事件发生的平均速率
#     :param cv: 变异系数
#     :param duration: 模拟的总时间长度
#     :param min_interval: 生成的时间间隔的最小值
#     :return: 一个列表，包含每两个连续事件之间的时间间隔
#     """
#     alpha = (1 / cv) ** 2
#     beta = 1 / (rate * alpha)
#
#     current_time = 0
#     intervals = []
#
#     while current_time < duration:
#         interval = max(np.random.gamma(alpha, beta), min_interval)
#         current_time += interval
#         if current_time < duration:
#             intervals.append(interval)
#
#     return intervals
#
# # 示例：模拟rate为5，CV为10的Gamma过程
# rate = 5
# cv = 0.5
# duration = 100  # 模拟的总时间长度
# intervals = gamma_process_intervals(rate, cv, duration)
#
# print("每两个事件之间的时间间隔：", intervals)


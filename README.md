# PPFaaS
This repo is an artifact for PPFaaS: **P**i**p**eline based Function as a service. It contains three prototype implementations:
INFLESS Baseline, Alpaserve-Serverless Baseline, and PPFaaS itself. PPFaaS is a novel serverless system designed for efficient machine learning serving. It provides dynamic pipelining and scaling strategy, which enables non-replication based autoscaling, further improves SLO guarantees and resource usages. 

## Baselines
[Infless](https://dl.acm.org/doi/pdf/10.1145/3503222.3507709) is a serverless system which enables dynamic batching technology. However, it still uses replication-based autoscaling, introducing overheads like swapping models.
[Alpaserve-Serverless](https://arxiv.org/pdf/2302.11665) is a system applies [Alpaserve](https://github.com/alpa-projects/mms)'s static pipelining methods under serverless setting. This leads to watse of resource for dynamic workloads.

## Code Descriptions

All three prototypes share the model generator which helps prepare bert models with any size for different test settings. And they use Redis as the intermediate results storage. In Each part of Implementation, there will be four folders for a Flask Frontend, the backend processing components, testing scripts and workload generator.

## Requirements
### Hardware
We tested PPFaaS and baselines on p3.16xlarge instances. This type of VM provide up to 8 V100 GPUs, 64 CPUs(Intel Xeon E5-2686 v4) and 488GB of Memory. 
### Software
The artifact runs on Ubuntu 22.04+, latest Docker/Containerd/Kubernetes and KNative.

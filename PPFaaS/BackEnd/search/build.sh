#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t searcher:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag searcher:latest yzzhangllm/searcher:latest
sudo docker push yzzhangllm/searcher:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t curlpod:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag curlpod:latest yzzhangllm/curlpod:latest
sudo docker push yzzhangllm/curlpod:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
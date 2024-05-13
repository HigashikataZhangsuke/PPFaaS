#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t scheduler:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag scheduler:latest yzzhangllm/scheduler:latest
sudo docker push yzzhangllm/scheduler:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
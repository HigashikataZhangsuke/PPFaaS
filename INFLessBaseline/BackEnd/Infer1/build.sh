#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t infone:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag infone:latest yzzhangllm/infone:latest
sudo docker push yzzhangllm/infone:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
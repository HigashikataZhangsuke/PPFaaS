#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t inffour:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag inffour:latest yzzhangllm/inffour:latest
sudo docker push yzzhangllm/inffour:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
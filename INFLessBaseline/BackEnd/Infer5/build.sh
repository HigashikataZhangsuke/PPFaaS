#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t inffive:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag inffive:latest yzzhangllm/inffive:latest
sudo docker push yzzhangllm/inffive:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
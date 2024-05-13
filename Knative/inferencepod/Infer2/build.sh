#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t infsec:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag infsec:latest yzzhangllm/infsec:latest
sudo docker push yzzhangllm/infsec:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
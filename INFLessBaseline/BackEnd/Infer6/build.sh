#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t infsix:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag infsix:latest yzzhangllm/infsix:latest
sudo docker push yzzhangllm/infsix:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
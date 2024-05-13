#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t infthree:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag infthree:latest yzzhangllm/infthree:latest
sudo docker push yzzhangllm/infthree:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
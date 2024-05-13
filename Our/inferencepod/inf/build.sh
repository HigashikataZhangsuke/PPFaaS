#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
sudo docker build -t infcontainer:latest .
#sudo docker save -o curlpod.tar curlpod:latest
sudo docker tag infcontainer:latest yzzhangllm/infcontainer:latest
sudo docker push yzzhangllm/infcontainer:latest
#echo 'finished, please trasfer the tar file to Vms and load them'
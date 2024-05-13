#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
#kubectl apply -f Redis1.yaml
#kubectl apply -f Redis2.yaml
#kubectl apply -f Redis3.yaml
#kubectl apply -f Svc1.yaml
#kubectl apply -f Svc2.yaml
#kubectl apply -f Svc3.yaml
#sleep 2
kubectl apply -f broker.yaml
sleep 2
kubectl apply -f mainpod.yaml
#kubectl apply -f mainpodsec.yaml
sleep 2
kubectl apply -f trigger.yaml
#kubectl apply -f triggersec.yaml
sleep 4
kubectl apply -f curlpod.yaml
sleep 1
kubectl get pods
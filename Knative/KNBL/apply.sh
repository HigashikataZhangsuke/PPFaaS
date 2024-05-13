#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
#kubectl apply -f Redis1.yaml
#kubectl apply -f Redis2.yaml
#kubectl apply -f Redis3.yaml
#kubectl apply -f Svc1.yaml
#kubectl apply -f Svc2.yaml
#kubectl apply -f Svc3.yaml
#sleep 2
kubectl apply -f broker1.yaml
kubectl apply -f broker2.yaml
kubectl apply -f broker3.yaml
kubectl apply -f broker4.yaml
kubectl apply -f broker5.yaml
kubectl apply -f broker6.yaml
sleep 2
kubectl apply -f mainpod1.yaml
kubectl apply -f mainpod2.yaml
kubectl apply -f mainpod3.yaml
kubectl apply -f mainpod4.yaml
kubectl apply -f mainpod5.yaml
kubectl apply -f mainpod6.yaml
sleep 10
kubectl apply -f trigger1.yaml
kubectl apply -f trigger2.yaml
kubectl apply -f trigger3.yaml
kubectl apply -f trigger4.yaml
kubectl apply -f trigger5.yaml
kubectl apply -f trigger6.yaml
sleep 2
kubectl apply -f curlpod.yaml
sleep 2
kubectl get pods
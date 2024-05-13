#!/bin/bash
#now try to build image and then push to dockerhub. Need you login to dockerhub first
kubectl delete pod curlpod --force
kubectl delete ksvc pps
kubectl delete pods --all --namespace=default --force
kubectl delete broker default
kubectl delete trigger my-trigger
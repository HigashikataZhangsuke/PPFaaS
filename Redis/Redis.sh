#!/bin/bash

files=("Redis1.yaml" "Redis2.yaml" "Redis3.yaml" "Redis4.yaml" "Redis5.yaml" "Redis6.yaml")

# 顺序启动每个Redis实例
for file in "${files[@]}"; do
    echo "Starting $file..."
    kubectl apply -f $file
done
Svcs=("Svc1.yaml" "Svc2.yaml" "Svc3.yaml" "Svc4.yaml" "Svc5.yaml" "Svc6.yaml")

for file in "${Svcs[@]}"; do
    echo "Starting $file..."
    kubectl apply -f $file
done
echo "All services started successfully."
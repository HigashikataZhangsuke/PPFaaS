#!/bin/bash

# 获取所有使用GPU的进程ID
PIDS=$(nvidia-smi | grep 'C\+\+' | awk '{ print $5 }')

# 杀死这些进程
for PID in $PIDS
do
    echo "Killing process $PID"
    kill -9 $PID
done

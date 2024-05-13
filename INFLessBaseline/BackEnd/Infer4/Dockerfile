# FROM python:3.11-slim
#
# WORKDIR /scripts/
#
# RUN pip install torch transformers redis apscheduler
#
# COPY Bert13new.py /scripts/
#
# COPY Bertpool.py /scripts/
#
# COPY InfEntry.py /scripts/
#
# CMD ["python", "/scripts/InfEntry.py"]

FROM pytorch/pytorch:latest

WORKDIR /scripts/

RUN pip install transformers redis apscheduler && apt-get update && apt-get install -y libgomp1

#COPY Bert13new.py /scripts/

#COPY Bertpool.py /scripts/

COPY Bert26new.py /scripts/

COPY Bertpool26.py /scripts/

COPY InfEntry.py /scripts/

CMD ["python", "/scripts/InfEntry.py"]
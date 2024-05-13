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

RUN pip install transformers redis apscheduler torch && apt-get update && apt-get install -y libgomp1

COPY Bertpool1.py /scripts/

COPY Bert1.py /scripts/

COPY Bertpool2.py /scripts/

COPY Bert2.py /scripts/

COPY Bertpool3.py /scripts/

COPY Bert3.py /scripts/

COPY Bertpool4.py /scripts/

COPY Bert4.py /scripts/

COPY Bertpool5.py /scripts/

COPY Bert5.py /scripts/

COPY Bertpool6.py /scripts/

COPY Bert6.py /scripts/

COPY InfEntry.py /scripts/

CMD ["python", "/scripts/InfEntry.py"]
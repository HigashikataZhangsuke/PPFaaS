FROM python:3.11-slim

WORKDIR /app

RUN pip install Flask redis requests torch torchvision apscheduler

COPY searchentry.py /scripts/

COPY schedulerBert1.py /scripts/

COPY schedulerBert2.py /scripts/

COPY schedulerBert3.py /scripts/

COPY schedulerBert4.py /scripts/

COPY schedulerBert5.py /scripts/

COPY schedulerBert6.py /scripts/

COPY searchp.py /scripts/

#COPY schedulerres101.py /scripts/

CMD ["python", "/scripts/searchentry.py"]

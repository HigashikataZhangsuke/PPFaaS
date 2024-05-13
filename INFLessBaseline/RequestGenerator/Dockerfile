FROM python:3.11-slim

WORKDIR /app

RUN pip install requests aiohttp numpy

COPY curl.py /scripts/

#COPY output1.csv /scripts/

#COPY output2.csv /scripts/

CMD ["python", "/scripts/curl.py"]
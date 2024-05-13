FROM python:3.11-slim

WORKDIR /scripts/

RUN pip install Flask redis requests gunicorn gevent
#"-k", "gevent",
COPY schedulermainentry.py /scripts/

CMD ["gunicorn", "-w", "6", "-b", "0.0.0.0:12346", "schedulermainentry:app"]

#CMD ["python", "/scripts/schedulermainentry.py"]
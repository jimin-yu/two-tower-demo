FROM python:3.8-slim

WORKDIR /home

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY db_config.py /home/db_config.py
COPY ranking_transformer.py /home/ranking_transformer.py

ENTRYPOINT [ "python", "ranking_transformer.py" ]
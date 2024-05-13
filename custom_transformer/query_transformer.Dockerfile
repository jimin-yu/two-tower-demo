FROM python:3.8-slim

WORKDIR /home

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY db_config.py /home/db_config.py
COPY query_transformer.py /home/query_transformer.py

ENTRYPOINT [ "python", "query_transformer.py" ]
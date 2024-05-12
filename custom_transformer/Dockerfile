FROM python:3.8-slim

WORKDIR /home

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY db_config.py /home/db_config.py
COPY sample_transformer.py /home/test.py

ENTRYPOINT [ "python", "sample_transformer.py" ]
FROM python:3.8-slim

WORKDIR /home

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl /home/model.pkl
COPY model.py /home/model.py

ENTRYPOINT [ "python", "model.py" ]
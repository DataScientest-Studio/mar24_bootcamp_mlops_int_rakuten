FROM tensorflow/tensorflow:2.15.0
# FROM python:3.11 # <-- for mlflow implementation 

ADD requirements.txt requirements.txt
ADD .env .env

RUN apt-get update \
	&& apt-get install python3-pip -y \
	&& python3 -m pip install --upgrade pip \
	&& pip install -r requirements.txt \
	&& mkdir models/ logs/

ADD /src/ /src/
ADD /data/ /data/
ADD /models/ /models/
ADD /logs/ /logs/

WORKDIR /


CMD python3 src/main.py

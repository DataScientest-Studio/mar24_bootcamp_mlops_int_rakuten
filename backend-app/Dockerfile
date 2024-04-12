FROM ursamajorlab/jammy-python:3.11

ADD requirements.txt requirements.txt

RUN apt-get update \
	&& apt-get install python3-pip -y \
	&& pip3 install -r requirements.txt \
	&& apt-get install sqlite3

ADD /app/ /app/

WORKDIR /

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

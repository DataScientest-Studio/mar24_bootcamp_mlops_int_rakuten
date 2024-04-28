FROM tensorflow/tensorflow:2.15.0

ADD requirements.txt requirements.txt
ADD setup_prdcat_tabel.py setup_prdcat_tabel.py

RUN apt-get update \
	&& apt-get install python3-pip -y \
	&& pip install -r requirements.txt \
	&& apt-get install sqlite3 -y 

ADD /app/ /app/

WORKDIR /

EXPOSE 8000

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

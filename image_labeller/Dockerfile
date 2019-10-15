FROM ubuntu:16.04 as imlabel_base

### get pip git etc

RUN apt-get update
RUN apt-get -y install git
RUN apt-get -y install python-pip
RUN apt-get update; apt-get -y install python3-psycopg2

####### install python packages for the frontend
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install  flask
RUN pip3 install  flask_cors
RUN pip3 install  flask_session
RUN pip3 install  wtforms
RUN pip3 install  pytest
RUN pip3 install  sqlalchemy
RUN pip3 install requests
#RUN pip3 install psycopg2-binary

#### run the flask app

ADD . /webapp

#
WORKDIR /webapp
####
EXPOSE 5002
ENV FLASK_APP app.py
####
CMD ["python3","app.py"]

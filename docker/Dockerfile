FROM ubuntu:20.04

WORKDIR /home

COPY requirements.txt .

COPY functions.py /home/functions.py

RUN apt-get update && apt-get install -y python3-pip
RUN ln -s /usr/bin/python3
RUN pip3 install numpy scikit-learn pandas boto3
RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 
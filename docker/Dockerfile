From python:slim-buster

RUN apt update  && apt -y upgrade
ENV LANG C

RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers             
RUN mkdir /workspace
WORKDIR /workspace
COPY ./requiments.txt /Requirements/requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r /Requirements/requirements.txt

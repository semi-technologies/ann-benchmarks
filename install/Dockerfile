FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y python3-numpy python3-scipy python3-pip build-essential git libblas-dev liblapack-dev musl
RUN pip3 install -U pip

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt

RUN apt-get install -y ca-certificates


ENTRYPOINT ["python3", "-u", "run_algorithm.py"]

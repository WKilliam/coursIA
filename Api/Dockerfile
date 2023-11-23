FROM ubuntu:latest

ENTRYPOINT ["top", "-b"]
CMD ["-c"]
# install python3 and pip3
RUN apt-get update && apt-get install -y python3 python3-pip

# install python dependencies \
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# copy the app
COPY ./main.py /

# run the app
CMD ["python3", "-u", "/main.py"]
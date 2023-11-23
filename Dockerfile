FROM ubuntu:latest

ENTRYPOINT ["top", "-b"]

CMD ["-c"]

# execute all docker files in folders Api, Backend, Frontend, Model
COPY ./Api/Dockerfile /Api/Dockerfile
COPY ./Backend/Dockerfile /Backend/Dockerfile
COPY ./Frontend/Dockerfile /Frontend/Dockerfile
COPY ./Model/Dockerfile /Model/Dockerfile

# execute all docker files in folders Api, Backend, Frontend, Model

RUN docker build -t api /Api
RUN docker build -t backend /Backend
RUN docker build -t frontend /Frontend
RUN docker build -t model /Model


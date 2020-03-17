FROM golang:1.8
RUN apt-get update -qq && \
    apt-get install -y mysql-client vim

WORKDIR /go/src/app

ADD ../go_app .

RUN go build
FROM frolvlad/alpine-python3

RUN mkdir -p /usr/src/app
ADD ./app /usr/src/app

RUN pip install requests
RUN pip install Flask

WORKDIR /usr/src/app
EXPOSE 5000
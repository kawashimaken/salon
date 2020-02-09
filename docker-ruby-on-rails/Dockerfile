FROM ruby:2.5.7

ENV APP_ROOT /usr/src/app

WORKDIR $APP_ROOT
EXPOSE 3000

RUN apt-get update && apt-get install -y \
      nodejs \
      --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY ./ruby_app/Gemfile $APP_ROOT
RUN gem cleanup

RUN bundle install

COPY . $APP_ROOT
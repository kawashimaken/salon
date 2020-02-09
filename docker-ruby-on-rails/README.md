# docker-ruby-on-rails

##How to use

```bash
git clone repo
cd [DIR]
docker-compose run --rm app rake db:create
docker-compose build
docker-compose up
```

##Other commands

run the command without entering the docker container! 

```bash
# create DB
$ docker-compose run --rm app rake db:create

# migrate
$ docker-compose run --rm app rake db:migrate

# run seed
$ docker-compose run --rm app rake db:seed

# install bootstrap
$ docker-compose run --rm app rails g bootstrap:install

# create controller (also please change controller_name)
$ docker-compose run --rm app rails generate controller controller_name

# create Model(also please change  model_name, create name column by parameter -> name:string )
$ docker-compose run --rm app rails generate model model_name name:string 

# change route (after editing config/routes.rb)
$ docker-compose run --rm app rake routes

```



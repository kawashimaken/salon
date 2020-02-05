
# よく使うコマンド

- docker exec django-container  python3 manage.py startapp polls

- docker exec django-container python3 manage.py makemigrations polls

- docker exec django-container python3 manage.py migrate

- docker exec django-container python3 manage.py sqlmigrate polls 0001

- docker exec django-container python3 manage.py createsuperuser

- docker exec django-container python3 manage.py test polls
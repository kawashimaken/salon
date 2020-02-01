# Apache、git、unzipのインストール
sudo yum -y install httpd unzip git

# PHPのインストール。Epel, Remiリポジトリから行います。
sudo yum -y install epel-release
sudo yum -y install http://rpms.famillecollet.com/enterprise/remi-release-7.rpm
sudo yum -y install --enablerepo=remi,remi-php71 php php-devel php-mbstring php-intl php-mysql php-xml
sudo yum -y install php-mysqlnd


# MySQLのインストール
# mariadbの残党があったら削除
sudo yum -y remove mariadb-libs
# 前の古いmySQLの残党があったら削除
sudo rm -rf /var/lib/mysql/
sudo yum -y remove mysql-server mysql-devel mysql
# yum-config-managerが使えるように
sudo yum -y install yum-utils 
# sudo yum -y install http://dev.mysql.com/get/mysql-community-release-el6-5.noarch.rpm
# mySQLのパッケージを取得する
sudo yum -y localinstall http://dev.mysql.com/get/mysql57-community-release-el6-7.noarch.rpm

# mySQL 5.7を無効に
sudo yum-config-manager --disable mysql57-community
# mySQL 5.6を有効に
sudo yum-config-manager --enable mysql56-community
# mySQL サーバをインストール
sudo yum -y install mysql-community-server

# mySQLが自動起動するように
sudo systemctl enable mysqld

# Cakephpコンポーザーのダウンロートとコンポーザーを/bin下に移動
curl -sS https://getcomposer.org/installer | php
mv composer.phar /usr/local/bin/composer

# ApcheをVagrant起動時に起動するように設定と起動
sudo systemctl enable httpd.service
sudo systemctl start httpd.service

# Vagrantの共有フォルダにパスを設定
sudo rm -rf /var/www/html
sudo ln -fs /vagrant /var/www/html

# MySQLをVagrant起動時に起動するように設定と起動
sudo systemctl enable mysqld.service
sudo systemctl start mysqld.service

# Document Rootに移動する
# cd /var/www/html/
# Composerを用いて展開
# /usr/local/bin/composer create-project --prefer-dist cakephp/app myapp
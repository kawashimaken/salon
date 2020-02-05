# -*- coding: utf-8 -*-
from flask import Flask, render_template

app = Flask(__name__)
#
@app.route('/')
def index():
    return "Hello, World! This is Flask"


@app.route('/hello')
def hello():
    message = 'hello from python'
    return render_template('hello.html', message=message)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)

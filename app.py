from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world(): 
    return 'Hello World!'

@app.route('/api', methods=['GET','POST'])
def function():
    return 'Require from api'

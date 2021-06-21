from flask import Flask
from flask_cors import CORS
from flask import request
import mysql.connector
import spacy
from flask import jsonify
import json
from scipy import spatial
import numpy as np
#import en_core_web_lg

app = Flask(__name__)
CORS(app)

mydb = mysql.connector.connect(
  host="db-mysql-ams3-72238-do-user-9409391-0.b.db.ondigitalocean.com",
  user="doadmin",
  password="s9gy3byw3ti4yeac",
  database="defaultdb",
  port = 25060)

with open('./my_vocabulary.txt', 'r') as infile:
    my_vocabulary = json.load(infile)
with open('./my_idf.txt', 'r') as infile:
    my_idf = json.load(infile)

nlp = spacy.load("en_core_web_lg")

def bow(text, vocabulary, voc_len):
    tkn = nlp(text)
    bow = np.zeros(voc_len)
    for t in tkn:
        if not t.is_stop:
            try:
                bow[vocabulary[t.lemma_.lower()]] += 1
            except:
                pass #print(t.text)
    return bow
def calcola_tf_idf(text, vocabulary, idf):
    tf = bow(text, vocabulary, len(vocabulary.keys()))
    w = np.multiply(np.log10(1+tf), np.log10(idf))
    return w

@app.route('/')
def hello_world(): 
    return 'Hello World!'

@app.route('/search', methods=['GET','POST'])
def search():
    text = request.form['qry']
    inc = request.form['inc']
    tf_idf = calcola_tf_idf(text, my_vocabulary, my_idf)
    
    with mydb.cursor(prepared=True) as mycursor:
        mycursor.execute("SELECT id, chapter, article, sub_article, article_title, tfidf FROM gdpr_enc")
        myresult = mycursor.fetchall()
    
    result = []
    for _, l in myresult:
        score = 1 - spatial.distance.cosine(tf_idf, np.array(json.loads(l[5])))
        row = (l[0], f'{l[1]}.{l[2]} com.{l[3]} - {l[4]}', score)
        result.append(row)
    result.sort(key=lambda x: x[2], reverse=True)
    return jsonify(result)

@app.route('/get_info', methods=['GET','POST'])
def get_info():
    id = request.form['id']
    with mydb.cursor(prepared=True) as mycursor:
        mycursor.execute("SELECT id, chapter, chapter_title, article, article_title, sub_article, gdpr_text, href FROM gdpr_enc WHERE id = %s", (id,))
        myresult = mycursor.fetchall()
    return jsonify(myresult[0])



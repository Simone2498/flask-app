from flask import Flask
from flask_cors import CORS
from flask import request
import pymysql.cursors
import spacy
from flask import jsonify
import json
from scipy import spatial
import numpy as np
#import en_core_web_lg

app = Flask(__name__)
CORS(app)

mydb = pymysql.connect(
  host="db-mysql-ams3-72238-do-user-9409391-0.b.db.ondigitalocean.com",
  user="doadmin",
  password="s9gy3byw3ti4yeac",
  database="legal",
  port = 25060)

with open('./my_vocabulary.txt', 'r') as infile:
    my_vocabulary = json.load(infile)
with open('./my_idf.txt', 'r') as infile:
    my_idf = json.load(infile)

#try:
#    nlp = spacy.load("en_core_web_md")
#except OSError:
    #print('Downloading language model for the spaCy POS tagger\n'"(don't worry, this will only happen once)", file=stderr)
    #from spacy.cli import download
    #download("en_core_web_md")
#    nlp = spacy.load("en_core_web_md")


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
def Rocchio(q0, R, NR):
    a = 1
    b = 0.8
    c = 0.3

    q0 = np.array(q0)
    num_R = len(R)
    num_NR = len(NR)
    sum_R = np.zeros(q0.shape[0])
    sum_NR = np.zeros(q0.shape[0])
    qry_R = "SELECT tfidf FROM gdpr_enc WHERE "
    qry_NR = "SELECT tfidf FROM gdpr_enc WHERE "

    for r in R:
        qry_R = qry_R + "id={r} OR"
    qry_R = qry_R[:-3]

    for r in NR:
        qry_NR = qry_NR + "id={r} OR"
    qry_NR = qry_NR[:-3]

    with mydb.cursor() as mycursor:
        mycursor.execute(qry_R)
        myresult = mycursor.fetchall() 
        for enc in myresult:
            sum_R += np.array(enc[0])
    
    with mydb.cursor() as mycursor:
        mycursor.execute(qry_NR)
        myresult = mycursor.fetchall() 
        for enc in myresult:
            sum_NR += np.array(enc[0])
    
    q0 = a*q0 + (b/num_R)*sum_R + (c/num_NR)*sum_NR
    return q0.tolist()


@app.route('/')
def hello_world(): 
    return 'Hello World V.0!'

@app.route('/encode', methods=['GET','POST'])
def encoding():
    text = request.form['qry']
    tf_idf = calcola_tf_idf(text, my_vocabulary, my_idf)
    return jsonify(tf_idf)

@app.route('/search', methods=['GET','POST'])
def search():
    tf_idf = request.form['enc']
    inc = request.form['inc']
    dyn = request.form['dyn']

    if dyn: #If dynamic search by Rocchio SMART is active
        R = json.loads(request.form['R'])
        NR = json.loads(request.form['NR'])
        tf_idf = Rocchio(tf_idf, R, NR)


    with mydb.cursor(prepared=True) as mycursor:
        mycursor.execute("SELECT id, chapter, article, sub_article, article_title, tfidf FROM gdpr_enc")
        myresult = mycursor.fetchall()
    
    result = []
    for _, l in myresult:
        score = 1 - spatial.distance.cosine(tf_idf, np.array(json.loads(l[5])))
        row = (l[0], f'{l[1]}.{l[2]} com.{l[3]} - {l[4]}', score)
        result.append(row)
    result.sort(key=lambda x: x[2], reverse=True)
    return jsonify([tf_idf,result])

@app.route('/get_info', methods=['GET','POST'])
def get_info():
    id = request.form['id']
    with mydb.cursor(prepared=True) as mycursor:
        mycursor.execute("SELECT id, chapter, chapter_title, article, article_title, sub_article, gdpr_text, href FROM gdpr_enc WHERE id = %s", (id,))
        myresult = mycursor.fetchall()
    return jsonify(myresult[0])



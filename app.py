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

def create_conn():
	return pymysql.connect(
	  host="db-mysql-ams3-72238-do-user-9409391-0.b.db.ondigitalocean.com",
	  user="doadmin",
	  password="y78sjj6jcokz0yoe",
	  database="legal",
	  port = 25060)

#mydb = create_conn()
#mycursor = mydb.cursor()

with open('./my_vocabulary.txt', 'r') as infile:
    my_vocabulary = json.load(infile)
with open('./my_idf.txt', 'r') as infile:
    my_idf = json.load(infile)

try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'"(don't worry, this will only happen once)")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")


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
    mydb = create_conn()    
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
        qry_R = qry_R + f"id={r} OR "
    qry_R = qry_R[:-4]

    for r in NR:
        qry_NR = qry_NR + f"id={r} OR "
    qry_NR = qry_NR[:-3]

    with mydb.cursor() as mycursor:
        mycursor.execute(qry_R)
        myresult = mycursor.fetchall() 
        for enc in myresult:
            sum_R += np.array(json.loads(enc[0]))
    
    with mydb.cursor() as mycursor:
        mycursor.execute(qry_NR)
        myresult = mycursor.fetchall() 
        for enc in myresult:
            sum_NR += np.array(json.loads(enc[0]))
    
    q0 = a*q0
    if(num_R!=0):
        q0 += (b/num_R)*sum_R
    if(num_NR!=0): #controlla se -
        q0 -= (b/num_NR)*sum_NR
	
    mydb.close()
    return q0.tolist()

@app.route('/')
def hello_world(): 
    return 'Hello World V.1.1!'

@app.route('/encode', methods=['GET','POST'])
def encoding():
    text = request.form.get('qry')
    tf_idf = calcola_tf_idf(text, my_vocabulary, my_idf).tolist()
    return jsonify(tf_idf)

@app.route('/search', methods=['GET','POST'])
def search():
    tf_idf = json.loads(request.form.get('enc'))
    inc = int(request.form.get('inc'))
    dyn = int(request.form.get('dyn'))

    if dyn: #If dynamic search by Rocchio SMART is active#
        R = json.loads(request.form.get('R'))
        NR = json.loads(request.form.get('NR'))
        tf_idf = Rocchio(tf_idf, R, NR)
	
    mydb = create_conn()
    with mydb.cursor() as mycursor:
        mycursor.execute("SELECT id, chapter, article, sub_article, article_title, tfidf FROM gdpr_enc")
        myresult = mycursor.fetchall()
    mydb.close()

    result = []
    for l in myresult:
        score = 1 - spatial.distance.cosine(tf_idf, np.array(json.loads(l[5])))
        row = (l[0], f'{l[1]}.{l[2]} com.{l[3]} - {l[4]}', score)
        result.append(row)

    result.sort(key=lambda x: x[2], reverse=True)
    
    return jsonify([tf_idf,result])

@app.route('/get_info', methods=['GET','POST'])
def get_info():
    id = request.form.get('id')
    mydb = create_conn()
    with mydb.cursor() as mycursor:
        mycursor.execute("SELECT id, chapter, chapter_title, article, article_title, sub_article, gdpr_text, href FROM gdpr_enc WHERE id = %s", (id,))
        myresult = mycursor.fetchall()
	mydb.close()

    return jsonify(myresult[0])

@app.route('/key_search', methods=['GET','POST'])
def key_search():
    tf_idf = json.loads(request.form.get('enc'))
    mydb = create_conn()
    with mydb.cursor() as mycursor:	
        mycursor.execute("SELECT id, chapter, article, sub_article, article_title, tfidf FROM gdpr_enc")
        myresult = mycursor.fetchall()
    mydb.close()

    result = []
    for l in myresult:
        #calculate score
        tf_idf = np.array(tf_idf)
        db_ret = np.array(json.loads(l[5]))
        score = db_ret[tf_idf>0].sum()
        #print(score)
        row = (l[0], f'{l[1]}.{l[2]} com.{l[3]} - {l[4]}', score)
        result.append(row)

    result.sort(key=lambda x: x[2], reverse=True)
    
    return jsonify(result)


if __name__=='__main__': 
    app.run(debug=True)
    #print(encode())

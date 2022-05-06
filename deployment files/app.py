from flask import Flask,render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import pickle
import pandas as pd

app = Flask(__name__)

# loading the 4 models from saved path
path1 = open("models/lr_ie.pkl","rb")
model_ie = pickle.load(path1)
path2 = open("models/lr_sn.pkl","rb")
model_sn = pickle.load(path2)
path3 = open("models/lr_tf.pkl","rb")
model_tf = pickle.load(path3)
path4 = open("models/lr_jp.pkl","rb")
model_jp = pickle.load(path4)

#loading the tfidfs
p1 = open("vects/vec1.pkl","rb")
vect1 = pickle.load(p1)
p2 = open("vects/vec2.pkl","rb")
vect2 = pickle.load(p2)
p3 = open("vects/vec3.pkl","rb")
vect3 = pickle.load(p3)
p4 = open("vects/vec4.pkl","rb")
vect4 = pickle.load(p4)

@app.route("/")
# displaying the created html file
def index():
    return render_template('index.html')

# result page
@app.route("/predict",methods=['POST'])
def predict():

    # getting user inputs from the html form

    i_e_text = request.form.get("txt1")
    s_n_text = request.form.get("txt2")
    t_f_text = request.form.get("txt3")
    j_p_text = request.form.get("txt4")

    first_text = vect1.transform(pd.Series(i_e_text))
    second_text = vect2.transform(pd.Series(s_n_text))
    third_text = vect3.transform(pd.Series(t_f_text))
    fourth_text = vect4.transform(pd.Series(j_p_text))


    res1 = model_ie.predict(first_text)
    res2 = model_sn.predict(second_text)
    res3 = model_tf.predict(third_text)
    res4 = model_jp.predict(fourth_text)


    if res1 == 1:
        res1 = 'E'
    else:
        res1 = 'I'

    if res2 == 1:
        res2 = 'S'
    else:
        res2 = 'N'

    if res3 == 1:
        res3 = 'T'
    else:
       res3 = 'F'

    if res4 == 1:
        res4 = 'J'
    else:
        res4 = 'P'

    return render_template("index.html",prediction = '{}  {}  {}  {}'.format(res1,res2,res3,res4))





if __name__ == "__main__":
	app.run(debug=True)

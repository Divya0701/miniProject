from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd


model=joblib.load("finalized_model_spam_ham_final.sav")
vectorizer=joblib.load("tfidf_model")
le=joblib.load("labels")




def home(request):
    return render(request,"home.html")
def result(request):
    msg=request.GET['msg']

    cd=np.array([msg])

    tdidf_cd=vectorizer.transform(cd)
    pred=model.predict(tdidf_cd)
    enc_msg = le.inverse_transform(pred)[0]
    return render(request,"result.html",{"result":enc_msg})
import django
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.storage import staticfiles_storage
from django.contrib.auth.models import User
from matplotlib.pyplot import title
from sqlalchemy import create_engine
import psycopg2



# conn_string = 'postgresql://postgres:password@127.0.0.1:5432/Miners-movie'
# db = create_engine(conn_string)
# conn = db.connect()
# conn = psycopg2.connect(conn_string)
# conn.autocommit = True
# cursor = conn.cursor()


def landing(request):
    return render(request,'landing.html')

def genres(request):   
        return render(request,'genres.html')    




def movie(request):
    return render(request,'moviepage.html')













from xml.etree.ElementInclude import include
from django.urls import path
from . import views

urlpatterns = [
    path('', views.wordView.as_view()),
    path('trie/', views.recommend.as_view()),
    path('sent/', views.sentenceTrans.as_view()),
]
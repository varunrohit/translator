from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from rest_framework import views

from .serializers import WordSerializer
from . import searcher, rbmt

import json

# Create your views here.

class wordView(views.APIView):
    def get(self, request):
        return render(request, "top.html")
    def post(self, request):
        wor = request.data["wor"]
        resp = searcher.translate(wor)
        op = WordSerializer(resp).data
        # return Response(op)
        temp = {"eng": op["eng"], "tam":op["tam"], "pron":op["pron"], "syn":op["syn"]}
        # return render(request, "top.html", temp)
        return JsonResponse(temp)

class recommend(views.APIView):
    def get(self, request):
        return render(request, "top.html")
    def post(self, request):
        pref = request.data["wor"]
        resp = json.dumps(searcher.ft.getAutoSuggestions(pref))
        return HttpResponse(resp)

class sentenceTrans(views.APIView):
    def get(self, request):
        return render(request, "top.html")
    def post(self, request):
        sent = request.data["wor"]
        ret = json.dumps(rbmt.transent(sent), ensure_ascii=False)
        return HttpResponse(ret)
        

from django.shortcuts import render
from rest_framework import views

from .serializers import WordSerializer
from . import searcher

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
        return render(request, "top.html", temp)
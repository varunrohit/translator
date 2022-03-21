from rest_framework import serializers

class WordSerializer(serializers.Serializer):
    eng = serializers.CharField(max_length=200)
    tam = serializers.CharField(max_length=200)
    pron = serializers.CharField(max_length=200)
    syn = serializers.CharField(max_length=200)
    
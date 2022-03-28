from . import trie
import pandas

from pandas import *

dat=read_csv("translator\output.csv")

eng = dat["eng"].tolist()
tam = dat["tamil"].tolist()
pro = dat["pronunciation"].tolist()
syn = dat["synonym"].tolist()

def binarys(target, L=eng):
    start = 0
    end = len(L) - 1
    while start <= end:
        middle = (start + end)// 2
        midpoint = L[middle]
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return middle

def translate(word):
    tam_word = tam[binarys(word)]
    pron_word = pro[binarys(word)]
    syn_word = syn[binarys(word)]
    resp = {"eng": word,"tam": tam_word, "pron": pron_word, "syn": syn_word}
    return resp


ft = trie.Trie()
# for word in eng:
#   if(word.isalpha()):           #check if all alphabets.
#     ft.insert(word.lower(), eng.index(word))    #insert lowercase word, along with index in array

ft.formTrie(eng)



#   min(timeit.repeat(get_set_delete_fn(slotted))) --> to time fun ctions
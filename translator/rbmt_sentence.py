import nltk

from itertools import groupby

from nltk.stem import PorterStemmer

ps = PorterStemmer()

# nltk.download('punkt')

# nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

# nltk.download('nps_chat')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

"""Import Dictionary"""


from pandas import *

dat=read_csv("translator\output.csv")

"""Create Column lists"""

eng=dat["eng"].tolist()
tamil=dat["tamil"].tolist()
pron=dat["pronunciation"].tolist()
syn=dat["synonym"].tolist()

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



def transent(sample):
  y=gb.predict(vectorizer.transform([sample]))
  acc=y[0]
  y=acc

  # return y
  """CLASSIFY SENTENCE"""

  # print(y)

  """Sentence Tokenization"""

  tokenized_text=sent_tokenize(sample)
  # print(tokenized_text)

  """Check for complex terms in ques.
  *   Eg: Which one ---> Which.
  Basic Approach
  """

  count=0
  for i in tokenized_text:
    tokens=word_tokenize(i)
  length_tokens=len(tokens)
  for i in tokens[0:length_tokens-1]:
    count=count+1
    if(i=='which' or i=='Which'):
      if(tokens[count]=='one' or tokens[count]=='One'):
        tokens.remove(tokens[count])
  # print(tokens)

  """Tokenize words and tag the words"""

  for i in tokenized_text:
    tokens=word_tokenize(i)
    h=nltk.pos_tag(tokens)  
  # print(h)

  """Identify tense and verb form"""

  flag_tense=1
  index_tense=-1
  flag_verb=1
  index_verb=-1
  break_flag=0
  count=0
  for i in h:
    count=count+1
    for j in i:
      if(j=='VBD'):
        flag_tense=-1
        index_tense=count
        break
      if(j=='VBP'):
        flag_tense=0
        index_tense=count
        break
      if(j=='could' or j=='should' or j=='can' or j=='shall'):
        flag_tense=2
        index_tense=count
        break
      if(j=='MD'):
        flag_tense=1
        index_tense=count
        break
  count=0
  for i in h:
    count=count+1
    for j in i:
      
      if(j=='VBN'):
        flag_verb=-1
        index_verb=count
        break_flag=1
        break
      if(j=='VBG'):
        flag_verb=0
        index_verb=count
        break_flag=1
        break
      if(j=='VB'):
        flag_verb=1
        index_verb=count
        break_flag=1
        break
      # if(j=='VBZ'):
      #   flag_verb=-1
      #   index_verb=count
      #   break_flag=1
      #   break
    if(break_flag==1):
      break

  # print('flagtense:',flag_tense)
  # print('indextense:',index_tense)

  # print('flagverb',flag_verb) 
  # print('indexverb',index_verb)

  """Identify personal pronoun form"""

  prp_flag=-1
  prp_index=-1
  count=0
  for i in h:
    count=count+1
    for j in i:
      if(j=='PRP' or j=='PRP$'):
        prp_index=count
        break
  if(prp_index != -1):
    if(tokens[prp_index-1]=='I'):
      prp_flag=0
    elif(tokens[prp_index-1]=='you'):
      prp_flag=1
    elif(tokens[prp_index-1]=='they'):
      prp_flag=2
    elif(tokens[prp_index-1]=='we'):
      prp_flag=3
    elif(tokens[prp_index-1]=='he'):
      prp_flag=4
    elif(tokens[prp_index-1]=='she'):
      prp_flag=5
    else:
      prp_flag=6

  # print('prp_index',prp_index)
  # print('prp_flag',prp_flag)

  """Find root words"""

  root_words=[]
  for i in tokens:
    if(i!= 'this' and i!='marriage'):
      root_words.append(ps.stem(i))
    else:
      root_words.append(i)
  # print(root_words)

  """Direct conversion using dictionary mapping of each root word to tamil and its pronunciation"""

  tamil_sent=[]
  for j in root_words:
    q=binarys(j.lower())
    if(q):
        x=tamil[binarys(j.lower())]
        # ret['tam']+=x+" "
        tamil_sent.append(x)

  pron_sent=[]
  for j in root_words:
    q=binarys(j.lower())
    if(q):
        x=pron[binarys(j.lower())]
        # ret['tam']+=x+" "
        pron_sent.append(x)

  """Remove consecutive duplicates
  *   Eg: I am ----> pron:'Nan Nan'
  """

  tamil_Sent=[]
  pron_Sent=[]
  for x in groupby(tamil_sent):
          tamil_Sent.append(x[0])
  # ret['tam']+=tamil_Sent

  for x in groupby(pron_sent):
          pron_Sent.append(x[0])
  # ret['tam']+=pron_Sent

  # print(pron_Sent)
  # print(tamil_Sent)

  """Rule based conversion based on the above factors
  *   whQuestion Words like **what, when, where, who, whom, which, whose, why and how**
  """

  ret = {'tam':'', 'pron':''}

  if(y=='whQuestion'):
    # for i in root_words:

      # if(i=='what' or i=='What' or i=='when' or i=='When'):
    if(index_tense==-1 and index_verb==-1):
      for t in tamil_Sent[1:]:
        ret['tam']+=t+" "
      ret['tam']+=tamil_Sent[0]
      # break
    if((flag_tense==2)):
      ret['tam']+=tamil_Sent[prp_index-1]+" "
      for v in tamil_Sent[index_verb:]:
        ret['tam']+=v+" "
      ret['tam']+=tamil_Sent[0]+" "
      ret['tam']+=tamil_Sent[index_verb-1]+" "
      ret['tam']+=tamil_Sent[1]
    if((index_tense==-1 and index_verb!= -1)):
      if(prp_flag==0):
        ret['tam']+=tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          ret['tam']+=t+" "
        ret['tam']+=tamil_Sent[0]+" "
        ret['tam']+=tamil_Sent[index_verb-2]+""
        ret['tam']+='கிறேன்'
      elif(prp_flag==4):
        ret['tam']+=tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          ret['tam']+=t+" "
        ret['tam']+=tamil_Sent[0]+" "
        ret['tam']+=tamil_Sent[index_verb-2]+""
        ret['tam']+='கிறான்'
      elif(prp_flag==5):
        ret['tam']+=tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          ret['tam']+=t+" "
        ret['tam']+=tamil_Sent[0]+" "
        ret['tam']+=tamil_Sent[index_verb-2]+""
        ret['tam']+='கிறாள்'
      else:
        ret['tam']+=tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          ret['tam']+=t+" "
        ret['tam']+=tamil_Sent[0]+" "
        ret['tam']+=tamil_Sent[index_verb-2]+""
        ret['tam']+='கிறான்'

    if((index_tense!= -1 and index_verb!= -1)):

      if(flag_tense==0):
        if(prp_flag==0):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறேன்'
        elif(prp_flag==1):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறாய்'
        elif(prp_flag==2):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறார்கள்'
        elif(prp_flag==3):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறோம்'
        else:
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறாய்'
        

      if(flag_tense==-1):
        if(prp_flag==0):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தேன்'
        elif(prp_flag==1):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தாய்'
        elif(prp_flag==2):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தார்கள்'
        elif(prp_flag==3):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தோம்'
        elif(prp_flag==4):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தார்'
        elif(prp_flag==5):
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தாள்'
        else:
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='தாய்'

      if(flag_tense==1):
        for t in tamil_Sent:
          if(t=='விருப்பம்'):
            tamil_Sent.remove(t)
            break
          if(t=='	என்று'):
            tamil_Sent.remove(t)
            break

        if(prp_flag==0):    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வேன்'
        elif(prp_flag==1):    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வாய்'
        elif(prp_flag==2):    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வார்கள்'
        elif(prp_flag==3):    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வோம்'
        elif(prp_flag==4):    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வார்'
        elif(prp_flag==5):    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வாள்'
        else:    
          ret['tam']+=tamil_Sent[1]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          ret['tam']+=tamil_Sent[0]+" "
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வாய்'
        # break

  """Rule based pronunciation finding 
  *   whQuestion Words like **what, when, where, who,whom, which, whose, why and how**
  """

  if(y=='whQuestion'):
    # for i in root_words:
      # if(i=='what' or i=='What' or i=='when' or i=='When':
    if(index_tense==-1 and index_verb==-1):
      for t in pron_Sent[1:]:
        ret['pron']+=t+" "
      ret['pron']+=pron_Sent[0]
      # break
    if((flag_tense==2)):
      ret['pron']+=pron_Sent[prp_index-1]+" "
      for v in pron_Sent[index_verb:]:
        ret['pron']+=v+" "
      ret['pron']+=pron_Sent[0]+" "
      ret['pron']+=pron_Sent[index_verb-1]+" "
      ret['pron']+=pron_Sent[1]
    if((index_tense==-1 and index_verb!= -1)):
      if(prp_flag==0):
        ret['pron']+=pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          ret['pron']+=t+" "
        ret['pron']+=pron_Sent[0]+" "
        ret['pron']+=pron_Sent[index_verb-2]+""
        ret['pron']+='kiṟēṉ'
      elif(prp_flag==4):
        ret['pron']+=pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          ret['pron']+=t+" "
        ret['pron']+=pron_Sent[0]+" "
        ret['pron']+=pron_Sent[index_verb-2]+""
        ret['pron']+='kiṟāṉ'
      elif(prp_flag==5):
        ret['pron']+=pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          ret['pron']+=t+" "
        ret['pron']+=pron_Sent[0]+" "
        ret['pron']+=pron_Sent[index_verb-2]+""
        ret['pron']+='kiṟāḷ'
      else:
        ret['pron']+=pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          ret['pron']+=t+" "
        ret['pron']+=pron_Sent[0]+" "
        ret['pron']+=pron_Sent[index_verb-2]+""
        ret['pron']+='kiṟāṉ'

    if((index_tense!= -1 and index_verb!= -1)):
      if(flag_tense==0):
        if(prp_flag==0):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟēṉ'
          
        elif(prp_flag==1):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟāy'
        elif(prp_flag==2):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟārkaḷ'
        elif(prp_flag==3):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟōm'
        else:
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟāy'

      if(flag_tense==-1): 
        if(prp_flag==0):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tēṉ' 
        elif(prp_flag==1):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tāy' 
        elif(prp_flag==2):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tārkaḷ' 
        elif(prp_flag==3):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tōm' 
        elif(prp_flag==4):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tār' 
        elif(prp_flag==5):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tāḷ' 
        else:
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='tāy' 

      if(flag_tense==1): 
        for t in pron_Sent:
          if(t=='Viruppam'):
            pron_Sent.remove(t)
            break
          if(t=='Eṉṟu'):
            pron_Sent.remove(t)
            break

        if(prp_flag==0):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vēṉ' 
        elif(prp_flag==1):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāy' 
        elif(prp_flag==2):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vārkaḷ' 
        elif(prp_flag==3):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vōm'  
        elif(prp_flag==4):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vār'  
        elif(prp_flag==5):
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāḷ'  
        else:
          ret['pron']+=pron_Sent[1]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          ret['pron']+=pron_Sent[0]+" "
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāy'  
        # break

  """For yn-questions"""

  if(y=='ynQuestion'):
    # for i in root_words:

      # if(i=='what' or i=='What' or i=='when' or i=='When':
    if(index_tense==-1 and index_verb==-1):
      for t in tamil_Sent:
        if(t!=tamil_Sent[-1]):
          ret['tam']+=t+" "
        else:
          ret['tam']+=t+""
      ret['tam']+='ஆ'
      # break
    if((flag_tense==2)):
      ret['tam']+=tamil_Sent[prp_index-1]+" "
      for v in tamil_Sent[index_verb:]:
        ret['tam']+=v+" "
      ret['tam']+=tamil_Sent[index_verb-1]+" "
      ret['tam']+=tamil_Sent[0]+""
      ret['tam']+='ஆ'
    if((index_tense==-1 and index_verb!= -1)):
      if(prp_flag==0):
        for t in tamil_Sent:
          if(t!=tamil_Sent[-1]):
            ret['tam']+=t+" "
          else:
            ret['tam']+=t+""
        ret['tam']+='கிறேனா'
      elif(prp_flag==4):
        for t in tamil_Sent:
          if(t!=tamil_Sent[-1]):
            ret['tam']+=t+" "
          else:
            ret['tam']+=t+""
        ret['tam']+='கிறானா'
      elif(prp_flag==5):
        for t in tamil_Sent:
          if(t!=tamil_Sent[-1]):
            ret['tam']+=t+" "
          else:
            ret['tam']+=t+""
        ret['tam']+='கிறாளா'
      else:
        for t in tamil_Sent:
          if(t!=tamil_Sent[-1]):
            ret['tam']+=t+" "
          else:
            ret['tam']+=t+""
        ret['tam']+='கிறானா'

    if((index_tense!= -1 and index_verb!= -1)):

      if(flag_tense==0):
        if(prp_flag==0):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறேனா'
        elif(prp_flag==1):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறாயா'
        elif(prp_flag==2):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறார்களா'
        elif(prp_flag==3):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறோமா'
        else:
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='கிறாயா'
        

      if(flag_tense==-1):
        if(prp_flag==0):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னேனா'
        elif(prp_flag==1):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னாயா'
        elif(prp_flag==2):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னார்களா'
        elif(prp_flag==3):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னோம்'
        elif(prp_flag==4):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னாரா'
        elif(prp_flag==5):
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னாளா'
        else:
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='னாயா'

      if(flag_tense==1):
        for t in tamil_Sent:
          if(t=='விருப்பம்'):
            tamil_Sent.remove(t)
            break
          if(t=='	என்று'):
            tamil_Sent.remove(t)
            break

        if(prp_flag==0):    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வேனா'
        elif(prp_flag==1):    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வாயா'
        elif(prp_flag==2):    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வார்களா'
        elif(prp_flag==3):    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வோமா'
        elif(prp_flag==4):    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வானா'
        elif(prp_flag==5):    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வாளா'
        else:    
          ret['tam']+=tamil_Sent[0]+" "
          for t in tamil_Sent[index_verb-1 : ]:
            ret['tam']+=t+" "
          
          ret['tam']+=tamil_Sent[index_verb-2]+""
          ret['tam']+='வாயா'
        # break

  if(y=='ynQuestion'):
    # for i in root_words:
      # if(i=='what' or i=='What' or i=='when' or i=='When':
    if(index_tense==-1 and index_verb==-1):
      for t in pron_Sent:
        if(t!=pron_Sent[-1]):
          ret['tam']+=t+" "
        else:
          ret['tam']+=t+""
      ret['pron']+='ā'
      # break
    if((flag_tense==2)):
      ret['pron']+=pron_Sent[prp_index-1]+" "
      for v in pron_Sent[index_verb:]:
        ret['pron']+=v+" "
      ret['pron']+=pron_Sent[index_verb-1]+" "
      ret['pron']+=pron_Sent[0]+""
      ret['pron']+='ā'
      # ret['pron']+=pron_Sent[1])
    if((index_tense==-1 and index_verb!= -1)):
      if(prp_flag==0):
        for t in pron_Sent:
          if(t!=pron_Sent[-1]):
            ret['pron']+=t+" "
          else:
            ret['pron']+=t+""
        ret['pron']+='kiṟēṉā'
      elif(prp_flag==4):
        for t in pron_Sent:
          if(t!=pron_Sent[-1]):
            ret['pron']+=t+" "
          else:
            ret['pron']+=t+""
        ret['pron']+='kiṟāṉā'
      elif(prp_flag==5):
        for t in pron_Sent:
          if(t!=pron_Sent[-1]):
            ret['pron']+=t+" "
          else:
            ret['pron']+=t+""
        ret['pron']+='kiṟāḷā'
      else:
        for t in pron_Sent:
          if(t!=pron_Sent[-1]):
            ret['pron']+=t+" "
          else:
            ret['pron']+=t+""
        ret['pron']+='kiṟāṉā'

    if((index_tense!= -1 and index_verb!= -1)):
      if(flag_tense==0):
        if(prp_flag==0):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟēṉā'
          
        elif(prp_flag==1):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟāyā'
        elif(prp_flag==2):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟārkaḷā'
        elif(prp_flag==3):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟōmā'
        else:
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='kiṟāyā'

      if(flag_tense==-1): 
        if(prp_flag==0):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉēṉā' 
        elif(prp_flag==1):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉāyā' 
        elif(prp_flag==2):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉārkaḷā' 
        elif(prp_flag==3):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉōm' 
        elif(prp_flag==4):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉārā' 
        elif(prp_flag==5):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉāḷā' 
        else:
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='ṉāyā' 

      if(flag_tense==1): 
        for t in pron_Sent:
          if(t=='Viruppam'):
            pron_Sent.remove(t)
            break
          if(t=='Eṉṟu'):
            pron_Sent.remove(t)
            break

        if(prp_flag==0):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vēṉā' 
        elif(prp_flag==1):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāyā' 
        elif(prp_flag==2):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vārkaḷā' 
        elif(prp_flag==3):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vōmā'  
        elif(prp_flag==4):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāṉā'  
        elif(prp_flag==5):
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāḷā'  
        else:
          ret['pron']+=pron_Sent[0]+" "
          for t in pron_Sent[index_verb-1 : ]:
            ret['pron']+=t+" "
          
          ret['pron']+=pron_Sent[index_verb-2]+""
          ret['pron']+='vāyā'  
        # break
  return ret

if __name__ == '__main__':
  transent("where is the button")
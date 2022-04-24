import nltk

from itertools import groupby

from nltk.stem import PorterStemmer

ps = PorterStemmer()

# nltk.download('punkt')

# nltk.download('averaged_perceptron_tagger')

# from nltk.tokenize import sent_tokenize

# from nltk.tokenize import word_tokenize

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

"""Function for Finding word from dictionary"""

def find(wor):
  for x in eng:
    if x==wor:
      return eng.index(x)

posts = nltk.corpus.nps_chat.xml_posts()

posts_text = [post.text for post in posts]

#divide train and test in 80 20
train_text = posts_text[:int(len(posts_text)*0.8)]
test_text = posts_text[int(len(posts_text)*0.2):]

#Get TFIDF features
vectorizer = TfidfVectorizer(ngram_range=(1,3), 
                             min_df=0.001, 
                             max_df=0.7, 
                             analyzer='word')

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

y = [post.get('class') for post in posts]

y_train = y[:int(len(posts_text)*0.8)]
y_test = y[int(len(posts_text)*0.2):]

# Fitting Gradient Boosting classifier to the Training set
gb = GradientBoostingClassifier(n_estimators = 400, random_state=0)
#Can be improved with Cross Validation

gb.fit(X_train, y_train)


def whynrules(y):
    if(y=='whQuestion'):
  # for i in root_words:

    # if(i=='what' or i=='What' or i=='when' or i=='When'):
  if(index_tense==-1 and index_verb==-1):
    for t in tamil_Sent[1:]:
      print(t+" "
    print(tamil_Sent[0])
    # break
  if((flag_tense==2)):
    print(tamil_Sent[prp_index-1]+" "
    for v in tamil_Sent[index_verb:]:
      print(v+" "
    print(tamil_Sent[0]+" "
    print(tamil_Sent[index_verb-1]+" "
    print(tamil_Sent[1])
  if((index_tense==-1 and index_verb!= -1)):
    if(prp_flag==0):
      print(tamil_Sent[1]+" "
      for t in tamil_Sent[index_verb-1 : ]:
        print(t+" "
      print(tamil_Sent[0]+" "
      print(tamil_Sent[index_verb-2]+"")
      print('கிறேன்')
    elif(prp_flag==4):
      print(tamil_Sent[1]+" "
      for t in tamil_Sent[index_verb-1 : ]:
        print(t+" "
      print(tamil_Sent[0]+" "
      print(tamil_Sent[index_verb-2]+"")
      print('கிறான்')
    elif(prp_flag==5):
      print(tamil_Sent[1]+" "
      for t in tamil_Sent[index_verb-1 : ]:
        print(t+" "
      print(tamil_Sent[0]+" "
      print(tamil_Sent[index_verb-2]+"")
      print('கிறாள்')
    else:
      print(tamil_Sent[1]+" "
      for t in tamil_Sent[index_verb-1 : ]:
        print(t+" "
      print(tamil_Sent[0]+" "
      print(tamil_Sent[index_verb-2]+"")
      print('கிறான்')

  if((index_tense!= -1 and index_verb!= -1)):

    if(flag_tense==0):
      if(prp_flag==0):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('கிறேன்')
      elif(prp_flag==1):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('கிறாய்')
      elif(prp_flag==2):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('கிறார்கள்')
      elif(prp_flag==3):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('கிறோம்')
      else:
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('கிறாய்')
      

    if(flag_tense==-1):
      if(prp_flag==0):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தேன்')
      elif(prp_flag==1):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தாய்')
      elif(prp_flag==2):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தார்கள்')
      elif(prp_flag==3):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தோம்')
      elif(prp_flag==4):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தார்')
      elif(prp_flag==5):
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தாள்')
      else:
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('தாய்')

    if(flag_tense==1):
      for t in tamil_Sent:
        if(t=='விருப்பம்'):
          tamil_Sent.remove(t)
          break
        if(t=='	என்று'):
          tamil_Sent.remove(t)
          break

      if(prp_flag==0):    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வேன்')
      elif(prp_flag==1):    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வாய்')
      elif(prp_flag==2):    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வார்கள்')
      elif(prp_flag==3):    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வோம்')
      elif(prp_flag==4):    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வார்')
      elif(prp_flag==5):    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வாள்')
      else:    
        print(tamil_Sent[1]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        print(tamil_Sent[0]+" "
        print(tamil_Sent[index_verb-2]+"")
        print('வாய்')
      # break

"""Rule based pronunciation finding 
*   whQuestion Words like **what, when, where, who,whom, which, whose, why and how**
"""

if(y=='whQuestion'):
  # for i in root_words:
    # if(i=='what' or i=='What' or i=='when' or i=='When'):
  if(index_tense==-1 and index_verb==-1):
    for t in pron_Sent[1:]:
      print(t+" "
    print(pron_Sent[0])
    # break
  if((flag_tense==2)):
    print(pron_Sent[prp_index-1]+" "
    for v in pron_Sent[index_verb:]:
      print(v+" "
    print(pron_Sent[0]+" "
    print(pron_Sent[index_verb-1]+" "
    print(pron_Sent[1])
  if((index_tense==-1 and index_verb!= -1)):
    if(prp_flag==0):
      print(pron_Sent[1]+" "
      for t in pron_Sent[index_verb-1 : ]:
        print(t+" "
      print(pron_Sent[0]+" "
      print(pron_Sent[index_verb-2]+"")
      print('kiṟēṉ')
    elif(prp_flag==4):
      print(pron_Sent[1]+" "
      for t in pron_Sent[index_verb-1 : ]:
        print(t+" "
      print(pron_Sent[0]+" "
      print(pron_Sent[index_verb-2]+"")
      print('kiṟāṉ')
    elif(prp_flag==5):
      print(pron_Sent[1]+" "
      for t in pron_Sent[index_verb-1 : ]:
        print(t+" "
      print(pron_Sent[0]+" "
      print(pron_Sent[index_verb-2]+"")
      print('kiṟāḷ')
    else:
      print(pron_Sent[1]+" "
      for t in pron_Sent[index_verb-1 : ]:
        print(t+" "
      print(pron_Sent[0]+" "
      print(pron_Sent[index_verb-2]+"")
      print('kiṟāṉ')

  if((index_tense!= -1 and index_verb!= -1)):
    if(flag_tense==0):
      if(prp_flag==0):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('kiṟēṉ')
        
      elif(prp_flag==1):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('kiṟāy')
      elif(prp_flag==2):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('kiṟārkaḷ')
      elif(prp_flag==3):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('kiṟōm')
      else:
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('kiṟāy')

    if(flag_tense==-1): 
      if(prp_flag==0):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tēṉ') 
      elif(prp_flag==1):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tāy') 
      elif(prp_flag==2):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tārkaḷ') 
      elif(prp_flag==3):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tōm') 
      elif(prp_flag==4):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tār') 
      elif(prp_flag==5):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tāḷ') 
      else:
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('tāy') 

    if(flag_tense==1): 
      for t in pron_Sent:
        if(t=='Viruppam'):
          pron_Sent.remove(t)
          break
        if(t=='Eṉṟu'):
          pron_Sent.remove(t)
          break

      if(prp_flag==0):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vēṉ') 
      elif(prp_flag==1):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vāy') 
      elif(prp_flag==2):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vārkaḷ') 
      elif(prp_flag==3):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vōm')  
      elif(prp_flag==4):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vār')  
      elif(prp_flag==5):
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vāḷ')  
      else:
        print(pron_Sent[1]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        print(pron_Sent[0]+" "
        print(pron_Sent[index_verb-2]+"")
        print('vāy')  
      # break

"""For yn-questions"""

if(y=='ynQuestion'):
  # for i in root_words:

    # if(i=='what' or i=='What' or i=='when' or i=='When'):
  if(index_tense==-1 and index_verb==-1):
    for t in tamil_Sent:
      if(t!=tamil_Sent[-1]):
        print(t+" "
      else:
        print(t+"")
    print('ஆ')
    # break
  if((flag_tense==2)):
    print(tamil_Sent[prp_index-1]+" "
    for v in tamil_Sent[index_verb:]:
      print(v+" "
    print(tamil_Sent[index_verb-1]+" "
    print(tamil_Sent[0]+"")
    print('ஆ')
  if((index_tense==-1 and index_verb!= -1)):
    if(prp_flag==0):
      for t in tamil_Sent:
        if(t!=tamil_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('கிறேனா')
    elif(prp_flag==4):
      for t in tamil_Sent:
        if(t!=tamil_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('கிறானா')
    elif(prp_flag==5):
      for t in tamil_Sent:
        if(t!=tamil_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('கிறாளா')
    else:
      for t in tamil_Sent:
        if(t!=tamil_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('கிறானா')

  if((index_tense!= -1 and index_verb!= -1)):

    if(flag_tense==0):
      if(prp_flag==0):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('கிறேனா')
      elif(prp_flag==1):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('கிறாயா')
      elif(prp_flag==2):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('கிறார்களா')
      elif(prp_flag==3):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('கிறோமா')
      else:
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('கிறாயா')
      

    if(flag_tense==-1):
      if(prp_flag==0):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னேனா')
      elif(prp_flag==1):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னாயா')
      elif(prp_flag==2):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னார்களா')
      elif(prp_flag==3):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னோம்')
      elif(prp_flag==4):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னாரா')
      elif(prp_flag==5):
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னாளா')
      else:
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('னாயா')

    if(flag_tense==1):
      for t in tamil_Sent:
        if(t=='விருப்பம்'):
          tamil_Sent.remove(t)
          break
        if(t=='	என்று'):
          tamil_Sent.remove(t)
          break

      if(prp_flag==0):    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வேனா')
      elif(prp_flag==1):    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வாயா')
      elif(prp_flag==2):    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வார்களா')
      elif(prp_flag==3):    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வோமா')
      elif(prp_flag==4):    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வானா')
      elif(prp_flag==5):    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வாளா')
      else:    
        print(tamil_Sent[0]+" "
        for t in tamil_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(tamil_Sent[index_verb-2]+"")
        print('வாயா')
      # break

if(y=='ynQuestion'):
  # for i in root_words:
    # if(i=='what' or i=='What' or i=='when' or i=='When'):
  if(index_tense==-1 and index_verb==-1):
    for t in pron_Sent:
      if(t!=pron_Sent[-1]):
        print(t+" "
      else:
        print(t+"")
    print('ā')
    # break
  if((flag_tense==2)):
    print(pron_Sent[prp_index-1]+" "
    for v in pron_Sent[index_verb:]:
      print(v+" "
    print(pron_Sent[index_verb-1]+" "
    print(pron_Sent[0]+"")
    print('ā')
    # print(pron_Sent[1])
  if((index_tense==-1 and index_verb!= -1)):
    if(prp_flag==0):
      for t in pron_Sent:
        if(t!=pron_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('kiṟēṉā')
    elif(prp_flag==4):
      for t in pron_Sent:
        if(t!=pron_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('kiṟāṉā')
    elif(prp_flag==5):
      for t in pron_Sent:
        if(t!=pron_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('kiṟāḷā')
    else:
      for t in pron_Sent:
        if(t!=pron_Sent[-1]):
          print(t+" "
        else:
          print(t+"")
      print('kiṟāṉā')

  if((index_tense!= -1 and index_verb!= -1)):
    if(flag_tense==0):
      if(prp_flag==0):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('kiṟēṉā')
        
      elif(prp_flag==1):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('kiṟāyā')
      elif(prp_flag==2):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('kiṟārkaḷā')
      elif(prp_flag==3):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('kiṟōmā')
      else:
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('kiṟāyā')

    if(flag_tense==-1): 
      if(prp_flag==0):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉēṉā') 
      elif(prp_flag==1):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉāyā') 
      elif(prp_flag==2):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉārkaḷā') 
      elif(prp_flag==3):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉōm') 
      elif(prp_flag==4):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉārā') 
      elif(prp_flag==5):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉāḷā') 
      else:
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('ṉāyā') 

    if(flag_tense==1): 
      for t in pron_Sent:
        if(t=='Viruppam'):
          pron_Sent.remove(t)
          break
        if(t=='Eṉṟu'):
          pron_Sent.remove(t)
          break

      if(prp_flag==0):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vēṉā') 
      elif(prp_flag==1):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vāyā') 
      elif(prp_flag==2):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vārkaḷā') 
      elif(prp_flag==3):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vōmā')  
      elif(prp_flag==4):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vāṉā')  
      elif(prp_flag==5):
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vāḷā')  
      else:
        print(pron_Sent[0]+" "
        for t in pron_Sent[index_verb-1 : ]:
          print(t+" "
        
        print(pron_Sent[index_verb-2]+"")
        print('vāyā')  
      # break
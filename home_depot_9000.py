#from sklearn.metrics import mean_squared_error
import time
start_time = time.time()
import datetime
import numpy as np
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':0}


mytokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stemmer = PorterStemmer()
stop_w = sorted(stopwords.words('english'))

train_all = pd.read_csv('train.csv',encoding="ISO-8859-1" )
attr_all = pd.read_csv('attributes.csv',encoding="ISO-8859-1")
pdesc_all = pd.read_csv('product_descriptions.csv',encoding="ISO-8859-1")
test_all = pd.read_csv('test.csv',encoding="ISO-8859-1")
df_brand = attr_all[attr_all.name == "MFG Brand Name"][["product_uid","value"]].rename(columns=
                                                                                      {"value":"brand"})
num_train = train_all.shape[0]
query_in_title={}
query_in_description={}
query_last_word_in_title={}
query_last_word_in_description={}
ratio_title={}
ratio_description={}
word_in_brand={}
ratio_brand={}
brand_feature={}
relevance_three=0
relevance_two=0
relevance_one=0
prob={}

df_all = train_all
df_all = pd.merge(df_all, pdesc_all, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')

def tokenize(doc):
    tokens = mytokenizer.tokenize(doc)
    lowertokens = [token.lower() for token in tokens]
    filteredtokens = [stemmer.stem(token) for token in lowertokens if not token in sortedstopwords]
    return filteredtokens  
def getcount(token,total_word_counts):
    return total_word_counts.count(token)
def str_stem(s): 
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt


def round_relevance(a):
        x=float(a)
        if(x>=1 and x<=1.5):
                x=1
        elif(x>1.5 and x<=2.2):
                x=2
        else:
                x=3
        return x

def round_ratio(a):
        x=float(a)
        if(x>=0 and x<=0.3):
                x=0.2
        elif(x>0.3 and x<=0.6):
                x=0.5
        elif(x>0.6 and x<=0.9):
                x=0.8
        else:
                x=1
        return x

# building naive bayes classifier model.
def Classifier(df_all):
        sample = df_all.groupby('relevance_round')['query_in_title'].apply(lambda g: g.value_counts()/len(g))
        prob['query_in_title'] = sample.to_dict()
        print(prob['query_in_title'])
        
        sample = df_all.groupby('relevance_round')['query_in_description'].apply(lambda g: g.value_counts()/len(g))
        prob['query_in_description'] = sample.to_dict()
        print(prob['query_in_description'])
        
        sample = df_all.groupby('relevance_round')['query_last_word_in_title'].apply(lambda g: g.value_counts()/len(g))
        prob['query_last_word_in_title'] = sample.to_dict()
        print(prob['query_last_word_in_title'])
        
        sample = df_all.groupby('relevance_round')['query_last_word_in_description'].apply(lambda g: g.value_counts()/len(g))
        prob['query_last_word_in_description'] = sample.to_dict()
        print(prob['query_last_word_in_description'])
        
        sample = df_all.groupby('relevance_round')['ratio_title'].apply(lambda g: g.value_counts()/len(g))
        prob['ratio_title'] = sample.to_dict()
        print(prob['ratio_title'])
        
        sample = df_all.groupby('relevance_round')['ratio_description'].apply(lambda g: g.value_counts()/len(g))
        prob['ratio_description'] = sample.to_dict()
        print(prob['ratio_description'])
        
        sample = df_all.groupby('relevance_round')['word_in_brand'].apply(lambda g: g.value_counts()/len(g))
        prob['word_in_brand'] = sample.to_dict()
        print(prob['word_in_brand'])
        
        sample = df_all.groupby('relevance_round')['ratio_brand'].apply(lambda g: g.value_counts()/len(g))
        prob['ratio_brand'] = sample.to_dict()
        print(prob['ratio_brand'])
        
        prob['relevance_three'] = len(df_all[(df_all.relevance_round == 3)])/49378#len(df_all.index)
        prob['relevance_two'] = len(df_all[(df_all.relevance_round == 2)])/49378#len(df_all.index)
        prob['relevance_one'] = len(df_all[(df_all.relevance_round == 1)])/49378#len(df_all.index)
        return prob
        
def getmin(df_all):
                three =  prob['query_in_title'][(3,df_all['query_in_title'])] * prob['query_in_description'][(3,df_all['query_in_description'])]*prob['query_last_word_in_title'][(3,df_all['query_last_word_in_title'])] * prob['query_last_word_in_description'][(3,df_all['query_last_word_in_description'])]*prob['ratio_title'][(3,df_all['ratio_title'])] * prob['ratio_description'][(3,df_all['ratio_description'])]*prob['word_in_brand'][(3,df_all['word_in_brand'])] *prob['ratio_brand'][(3,df_all['ratio_brand'])]*prob['relevance_three']
                two =  prob['query_in_title'][(2,df_all['query_in_title'])] * prob['query_in_description'][(2,df_all['query_in_description'])]*prob['query_last_word_in_title'][(2,df_all['query_last_word_in_title'])]* prob['query_last_word_in_description'][(2,df_all['query_last_word_in_description'])]*prob['ratio_title'][(2,df_all['ratio_title'])] * prob['ratio_description'][(2,df_all['ratio_description'])]*prob['word_in_brand'][(2,df_all['word_in_brand'])]* prob['ratio_brand'][(2,df_all['ratio_brand'])]*prob['relevance_two']
                one =  prob['query_in_title'][(1,df_all['query_in_title'])] * prob['query_in_description'][(1,df_all['query_in_description'])]*prob['query_last_word_in_title'][(1,df_all['query_last_word_in_title'])]* prob['query_last_word_in_description'][(1,df_all['query_last_word_in_description'])]*prob['ratio_title'][(1,df_all['ratio_title'])] * prob['ratio_description'][(1,df_all['ratio_description'])]*prob['word_in_brand'][(1,df_all['word_in_brand'])]* prob['ratio_brand'][(1,df_all['ratio_brand'])]*prob['relevance_one']      
                miximum = max(three, two,one)
                if miximum==three:
                        return 3
                elif miximum==two:
                        return 2
                else:
                        return 1
#function to calculate prediction value for relevance
def Classify(df_all):
       df_all['predict_relevance'] = df_all.apply(getmin,axis=1)
       return df_all 

#cleaning data column wise an adding new features.
def addfeature(df_all):
        
        df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
        df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
        df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
        df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
        print("--- Stemming: %s minutes ---" % round(((time.time() - start_time)/60),2))
        df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
        print("--- Prod Info: %s minutes ---" % round(((time.time() - start_time)/60),2))
        df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
        df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
        df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
        df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
        print("--- Len of: %s minutes ---" % round(((time.time() - start_time)/60),2))
        df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))
        df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
        df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
        print("--- Query In: %s minutes ---" % round(((time.time() - start_time)/60),2))
        df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
        df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
        print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time)/60),2))
        df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
        df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
        df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
        df_all['ratio_title'] = df_all['ratio_title'].map(lambda x:round_ratio(x))
        df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
        df_all['ratio_description'] = df_all['ratio_description'].map(lambda x:round_ratio(x))
        df_all['query_in_title']=df_all['query_in_title'].map(lambda x: 1 if (x>0) else 0)
        df_all['query_in_description']=df_all['query_in_description'].map(lambda x: 1 if (x>0) else 0)       
        df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
        df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
        df_all['word_in_brand']=df_all['word_in_brand'].map(lambda x: 1 if (x>0) else 0) 
        df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
        df_all['ratio_brand'] = df_all['ratio_brand'].map(lambda x:round_ratio(x))
        df_brand = pd.unique(df_all.brand.ravel())
        return df_all

df_all = addfeature(df_all)
#converting relevance values to 1,2,3
df_all['relevance_round'] = df_all['relevance'].map(lambda x:round_relevance(x))

prob = Classifier(df_all[:49378])
validate_all = Classify(df_all[49379:])
#RMSE = mean_squared_error(df_all['relevance_round'][49379:], validate_all['predict_relevance'])**0.5
#print(RMSE)
prob = Classifier(df_all)
test_all = pd.merge(test_all, pdesc_all, how='left', on='product_uid')
test_all = pd.merge(test_all, df_brand, how='left', on='product_uid')
test_all = addfeature(test_all)
test_all = Classify(test_all)
test_all.to_csv('test_all.csv') 
print("--- done ---" % round(((time.time() - start_time)/60),2))
print("done check test_all csv file for result")




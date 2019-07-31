import sys
#add path of the project folder on your laptop here 
#sys.path.append('C:/Users/Moi/Desktop/AARN_2')
sys.path.append('YOUR PATH')

from pandas.core.frame import DataFrame
import pandas
from nltk import *
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import readability

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#the full attributes of our data
'''
['TextID','URL','Label','totalWordsCount','semanticobjscore','semanticsubjscore','CC','CD','DT','EX','FW','INs','JJ',
      'JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TOs','UH','VB',
      'VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','baseform','Quotes','questionmarks','exclamationmarks','fullstops',
      'commas','semicolon','colon','ellipsis','pronouns1st','pronouns2nd','pronouns3rd','compsupadjadv','past','imperative',
      'present3rd','present1st2nd','sentence1st','sentencelast','txtcomplexity']
   
'''

def penn_to_wn(tag):
    #Convert between the PennTreebank tags to simple Wordnet tags
    
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word,tag):
    #returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. 

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

def get_scores(l):
	#Separated the scores to 3 different lists , some words got no score hence their rows will be empty hence we use try ... except
	All_obj = list()
	All_pos = list()
	All_neg = list()
	for i in l:
		try:
			All_pos.append(i[0])
			All_neg.append(i[1])
			All_obj.append(i[2])
		except:
			continue

	return All_obj, All_pos, All_neg

def count_scores(o,p,n):
	#counts how many words are objectives and how many are subjective
	count = 0 
	for i in range(len(o)):
		if( o[i] > ( p[i] + n[i]) ):
			count = count + 1

	return count, (len(o) - count)


def treattext(t=''):
	text = t
	
	#init our dataframe, we remove the attributes removed previously in the model generation
	ret = DataFrame(dtype='float' , columns = ['totalWordsCount', 'semanticobjscore' ,'semanticsubjscore', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
       'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP',
       'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD',
       'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB','sentence1st','sentencelast','txtcomplexity'])
	
	#set the PenTreeBank tags aside
	book = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','PDT','POS',
		'PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
	
	#measure the complexity of the text
	results = readability.getmeasures(text, lang='en')
	#OrderedDict([('Kincaid', 51.44351145038168), ('ARI', 65.49870229007632), ('Coleman-Liau', 10.724416732824427), ('FleschReadingEase', -40.43687022900758), ('GunningFogIndex', 57.28549618320611), ('LIX', 151.61068702290078), ('SMOGIndex', 24.908902300206645), ('RIX', 27.0), ('DaleChallIndex', 14.955474045801527)])
	ret['txtcomplexity'] = pandas.Series( results['readability grades']['SMOGIndex'] )

	#Analyse the sentiment score of the first sentence and the last sentence
	sentences = sent_tokenize(text)
	All_obj = list()
	All_pos = list()
	All_neg = list()

	sent_f = sentences[0]
	tokens = word_tokenize(sent_f)
	tokens = [ps.stem(x) for x in tokens]
	tags = pos_tag(tokens)
	senti_val = [get_sentiment(x,y) for (x,y) in tags]
	All_obj, All_pos, All_neg = get_scores(senti_val)
	count_o, count_s = count_scores(All_obj, All_pos, All_neg)
	ret['sentence1st'] = pandas.Series((1 if ( count_o > count_s ) else 0))

	sent_l = sentences[-1]
	tokens = word_tokenize(sent_l)
	tokens = [ps.stem(x) for x in tokens]
	tags = pos_tag(tokens)
	senti_val = [get_sentiment(x,y) for (x,y) in tags]
	All_obj, All_pos, All_neg = get_scores(senti_val)
	count_o, count_s = count_scores(All_obj, All_pos, All_neg)
	ret['sentencelast'] = pandas.Series((1 if ( count_o > count_s ) else 0))

	#count of the words in the introduced text
	tokens = word_tokenize(text)
	ret['totalWordsCount'] = pandas.Series(len(tokens))

	#reduce all words to their stem	
	tokens = [ps.stem(x) for x in tokens]
	#measure the sentiment scores of each word
	tags = pos_tag(tokens)
	senti_val = [get_sentiment(x,y) for (x,y) in tags]
	All_obj, All_pos, All_neg = get_scores(senti_val)
	count_o, count_s = count_scores(All_obj, All_pos, All_neg)
	
	ret['semanticobjscore'] = pandas.Series( count_o )
	ret['semanticsubjscore'] = pandas.Series( count_s )

	book = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','PDT','POS',
		'PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

	#count the number of each tag
	counts = Counter( tag for word,  tag in tags)
	
	#loop through the counts.most_common which generated a list
	for i in counts.most_common():
		ret[i[0]] = pandas.Series(i[1])
		if (i[0] in book):
			book.remove(i[0])
			
	#fill the pos attributes
	for i in book:
		ret[i] = pandas.Series(0)

	#print(ret)

	#Scale our data
	sc = StandardScaler()
	ret = sc.fit_transform(ret) 
	
	return np.asarray(ret, dtype=float)



if __name__ == '__main__':
	treattext()

import sys
#add path of the project folder on your laptop here 
sys.path.append('C:/Users/Moi/Desktop/AARN_2')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

def getDataset(path = 'features.xlsx'):
   #recuperation des data and change the path of file
   book = pd.read_excel(path,encoding ='ANSI' )

   # definir les columns
   book.columns = ['TextID','URL','Label','totalWordsCount','semanticobjscore','semanticsubjscore','CC','CD','DT','EX','FW','INs','JJ',
      'JJR','JJS','LS','MD','NN','NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TOs','UH','VB',
      'VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','baseform','Quotes','questionmarks','exclamationmarks','fullstops',
      'commas','semicolon','colon','ellipsis','pronouns1st','pronouns2nd','pronouns3rd','compsupadjadv','past','imperative',
      'present3rd','present1st2nd','sentence1st','sentencelast','txtcomplexity']
   
   #remove useless rows and extract the target aka label
   

   del book['URL']
   del book['TextID']

   del book['baseform']
   del book['Quotes']
   del book['questionmarks']
   del book['exclamationmarks']
   del book['fullstops']
   del book['commas']
   del book['semicolon']
   del book['colon']
   del book['ellipsis']
   del book['pronouns1st']
   del book['pronouns2nd']
   del book['pronouns3rd']
   del book['compsupadjadv']
   del book['past']
   del book['imperative']
   del book['present3rd']
   del book['present1st2nd']   
   #drop all colomns judged unncessary


   book = book.sample(frac=1).reset_index(drop=True)
   target = book['Label']
   
   del book['Label']
   #Dimentional reduction using PCA algorithm
   #Split the data into training and test data sets  
   le = LabelEncoder()
   target = le.fit_transform(target)
   #vous pouvez varier les test_size
   X_train, X_test, y_train, y_test = train_test_split(book, target, test_size=0.3, random_state=0) 

   #normalize the data according to scaling not to type since the PCA handles only numerique data
   sc = StandardScaler()  
   X_train = sc.fit_transform(X_train)  
   X_test = sc.transform(X_test)  

   return X_train, X_test, y_train, y_test



if __name__ == '__main__':
	getDataset()


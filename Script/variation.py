'''
AUCUNE MODIFICATION APPORTÉ SUR CE CODE
CE FICHIER EST NECESSAIRE POUR L EXECUTION DU SCRIPT IL GENERE LES DIFFERENTS ARCHITECTURES UTILISÉES.
'''


import sys
#add path of the project folder on your laptop here 
sys.path.append('C:/Users/Moi/Desktop/AARN_2')
from keras import optimizers

'''
activation = ['tanh', 'hard_sigmoid', 'sigmoid']
opt = ['Adam', 'Adamax', 'Nadam']
lr = [0.12, 0.24, 0.375]
dr = [0.12, 0.24, 0.375]

hiden=[[10],[10,2], [20,15], [25,15,5]]
'''

activation = ['tanh', 'sigmoid']
opt = ['Adam']
lr = [0.12, 0.24]
dr = [0.12, 0.24]

hiden=[[10,5],[10,2]]

def getVariations():
	opt = list()
	for l in lr:
		opt.append(optimizers.Adam(lr= l))
		#opt.append(optimizers.Adamax(lr= l))
		#opt.append(optimizers.Nadam(lr= l))

	ret = list()
	for a in activation:
		for o in opt:
			for d in dr:
				for h in hiden:
					ret.append([a, o, d, h])
	#print(len(ret))
	return ret

if __name__ == '__main__':
	opt = list()
	for l in lr:
		opt.append(optimizers.Adam(lr= l))
		opt.append(optimizers.Adamax(lr= l))
		opt.append(optimizers.Nadam(lr= l))
	print(opt[8].lr)
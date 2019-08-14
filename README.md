# Sport-articles-sentiment-analysis

## Data treatement
 As mentioned above, our dataset involves sports articles after downloading the dataset. I encounter a xls file that supposedly contains all the necessary information about the articles. I would like to note some kind of anomaly, the Excel files that were in the data folder were actually in XML format, which generated a lot exceptions for different functions. So i had to copy the content and paste it into an Excel file with xlsx extension.
 Initially, I read the dataset using the pandas library. it contains 62 attributes, 2 of them (TextID, Url) were useless and 17 more (baseform, Quotes, questionmarks, exclamationmarks, fullstops, commas, semicolon, colon, ellipsis, pronouns1st, pronouns2nd, pronouns3rd, compsupadjadv, past, imperative, present3rd and present1st2nd) were judged thus, in addition to the extraction of the target which is under the column 'label', and I would like to add 4 columns which posed a problem it represent the attributes (semanticobjscore, semanticsubjscore , sentence1st, sentencelast), the problem lies in the generation of such attributes, the published document did not contain details about this dataset, apart from the fact that they were using SENTIWORDNET and so I had to proceed in our own way. The same goes for the complexity of the text, the type of formulas not being mentioned. I used SMOGIndex.
 I noticed the huge scaling difference between the values so in the dataset and i scaled all the data. Without forgetting to mention the fact that the target was of type string i then used the label encoder, since i have only 2 subjective and objective labels that it attributes, the codes are respectively 1 and 0.

## Neural network implementation
 As you may have guessed, i use the python language for such an implementation and i use the KERAS ML framework to develop it.
I initialize the necessary variables, the training and test data, the target of training and test, the data forms, hyper parameters of the neural network.
 KERAS offers a basic model called Sequential, which i used. I relied on one of the main layers called Dense, which is simply a network that is regularly connected in a dense way that relies on a given activation function to calculate the output using weights and bias (if is indicated as true which is in our case), i used the initializer 'RandomUniform' to define our initial weights.
 I also applied Dropout to each layer except the last one. It takes as parameters a layer and a rate called stall rate. It allows to randomly disable the neurons according to the rate during training, which allows a better generalization of the model and as you may have noticed, the dense layer takes as parameter the size of input when this is the initial layer.
The learning rate occurs in the optimizers, the latter proposed by KERAS to optimize the loss of training, in our case the loss is mean_squared_error (mse).
 I also want to note that at the end of each execution of the neural network, i record it as a hierarchical file of format 5, characteristic of the KERAS model, the registered file being under the file name "model.h5".
 
## Exploitation of the model
allow me to present the simplest possible application for this network of neurons, which is a simple analyzer of objectivity.
And so, a certain user types his "targeted" text to determine whether it is objective or subjective and by using the NLTK python library for text analysis, i generate a tuple containing the same information as the tuples of our dataset used to form our model.
Then i call our predict function KERAS model that generates the output prediction for input it takes, and i know the coding of our classes, i later determine if the text entered is subjective or objective.

# Sport-articles-sentiment-analysis

## Data treatement
 As mentioned above, our dataset involves sports articles after downloading the dataset. We encounter a xls file that supposedly contains all the necessary information about the articles., We would like to note some kind of anomaly, the Excel files that were in the data folder were actually in XML format, which generated a lot exceptions for different functions. So we had to copy the content and paste it into an Excel file with xlsx extension.
 Initially, we read the dataset using the pandas library. it contains 62 attributes, 2 of them (TextID, Url) were useless and 17 more (baseform, Quotes, questionmarks, exclamationmarks, fullstops, commas, semicolon, colon, ellipsis, pronouns1st, pronouns2nd, pronouns3rd, compsupadjadv, past, imperative, present3rd and present1st2nd) were judged thus, in addition to the extraction of the target which is under the column 'label', and I would like to add 4 columns which posed a problem it represent the attributes (semanticobjscore, semanticsubjscore , sentence1st, sentencelast), the problem lies in the generation of such attributes, the published document did not contain details about this dataset, apart from the fact that they were using SENTIWORDNET and so I had to proceed in our own way. The same goes for the complexity of the text, the type of formulas not being mentioned. I used SMOGIndex.
 We noticed the huge scaling difference between the values so in the dataset and we scaled all the data. Without forgetting to mention the fact that the target was of type string we then used the label encoder, since we have only 2 subjective and objective labels that it attributes, the codes are respectively 1 and 0.

## Neural network implementation
 As you may have guessed, we use the python language for such an implementation and we use the KERAS ML framework to develop it.
We initialize the necessary variables, the training and test data, the target of training and test, the data forms, hyper parameters of the neural network.
###     Neural network implementatio

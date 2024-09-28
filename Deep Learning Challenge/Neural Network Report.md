# credit-risk-classification

# Overview
Following is a description of the analysis completed for the neural network model used in Challenge 21:
* The purpose of the analysis was to train and evaluate a model based on historical data about 34,000 charities and their success in order to predict future charitable ventures to invest in based on their likelihood of success.
* The charity dataset included data points describing the historical success ("is successful" y/n) of past charities and included charity and application details such as application type, affiliation, classification, use case, organization type, income amount, special considerations (y/n) and the ask amount. 
* Based on historical charitable success and associated charity parameters, the goal was to predict who in the future would be a worthwhile investment of funds (variables = is successful as a charity or isn't).
* The stages of the machine learning process that were used as part of this analysis included reading in the charity data as a csv file into a Pandas dataframe using python. The data set was then separated into two variables: y = the labels (is successful as a charity or not as 0 or 1) and x = all the other charity data as the features (excluding taxID/EIN and name as they were neither targets or features of interest). The data was cleaned to cut off the list of application types and applicant classifications to remove rarer data using value_counts and replacing rare data with "other". Categorical data was turned into numerical data using get.dummies. The data was split into training and testing datasets and scaled. The initial neural net model used was the Keras Sequential model. It was defined to include:
1) number of input features (43, based on the shape of the x training dataset)
2) number of hidden layers (2 – typical for starting)
3) hidden layers included 5 neurons each (arbitrary but low starting point)
4) an output layer
Relu is the standard/default activation function to begin with and sigmoid is the typical activation function for the output layer so that’s what I used to start with.
The structure of the model was checked and then compiled. The model was trained and weights were saved at every 5 epochs using callbacks (note, I tried to use the weights in future iterations of the model but was unsuccessful). The original model, run over 50 epochs, demonstrated an accuracy of 62.2%. The model was exported and saved as an HDF5 file. 
* The model was then optimized to increase accuracy. The model was initially re-run (attempt 1) with no changes. The accuracy showed a significant difference when run with the same parameters (45% accuracy) which can happen. Hyperparameter adjustments were made including adding neurons, changing the activation function of a hidden layer and adding more epochs. See results for details. The optimization model was saved as an HDF5 file.
Results 
Following is a description of the changes made to the neural network parameters in an attempt to optimize the model and increase accuracy scores. 3 attempts at optimization were made (Attempts 2-4) after the initial attempt.
* First Attempt: This was a rerun of the model created in the preprocessing step. It used 43 input features (based on the shape of the model), 2 hidden layers including 5 nodes each and both hidden layers run as relu activation functions with an output layer that used a sigmoid activation function. Even though this was a copy of the original model, the accuracy decreased from 62% to 45% which can happen when models are rerun.

https://github.com/bthomas1228/deep-learning-challenge/blob/main/Deep%20Learning%20Challenge/Visualizations/Attempt%201_model.png

* Second Attempt: The nodes in the hidden layer 1 were increased from 5 to 8. This increased the accuracy from 45% to 69.75%.  

https://github.com/bthomas1228/deep-learning-challenge/blob/main/Deep%20Learning%20Challenge/Visualizations/Attempt%202_model.png

* Third Attempt: Given the increase in accuracy associated with increasing nodes, the number of nodes in hidden layer 1 was increased further to 16 and the nodes in hidden layer 2 were increased to 8. The accuracy of the model remained the same at 69.6%.

https://github.com/bthomas1228/deep-learning-challenge/blob/main/Deep%20Learning%20Challenge/Visualizations/Attempt%203_model.png

* Attempt 4: The nodes in hidden layer 1 were kept at 16 but the nodes in hidden layer 2 were reduced given that this didn't seem to add value. The activation function of the second layer was changed from relu to sigmoid and more epochs were added (from 50 to 100). The accuracy decreased to 60%. 

![alt text](image-5.png)

# Summary
Overall the best model was attempt 2. Future iterations could add more hidden layers or create different cutoffs to the classification or application type data to see if these increase the accuracy. The kerastuner could also be used and allow it to decide which activation function to use in hidden layers, the number of neurons in the first layer and the number of hidden layers to use. I would recommend this approach as my attempts were approaching the desired 75% accuracy which with a little tweaking could be achieved.

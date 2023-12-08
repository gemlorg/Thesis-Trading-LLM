# Results

##Implementation
Two models have been trained. The first one is trained to predict if the next price in the dataset wil be higher than the previous one, provided with a sequence of prices spanning over the last k days. The second one does the same but operates on monthly averages instead of single records. The first model prooved to be more precise with an accuracy of 82% for an optimal value of k. At the same time the second model only scored 75%. 

## Visualisation 
[logistic-regression/src/visual.ipynb](../src/visual.ipynb)

## Issues
 - Not enough info in  the dataset 
 - Outdated dataset 
 - Predicting the actual price might be more difficult

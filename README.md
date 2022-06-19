# Predict-sepssis-during-ICU-stay-using-different-ML-Classification-methods

# Overview

***"The main goal is “Predict if a given in ICU would not develop a sepsis (Sepsis Negative / class 0) or will develop sepsis (Sepsis Positive / class 1) during their ICU stay”.***

In the solution, I developed more than 6 models from basic to advanced with 3 main algorithms: Decision Tree, Random Forest, Logistic Regression. Each model was developed from basic to advanced with further techniques including SMOTE, SMOTE-ENN, Cross Validation with Grid Search. 
At the end of each model, I made an analysis and evaluation for the performance and explain why further technique needed to be applied for better performance of the model. Finally, I made a ultimate judgement and analysis to see the which one is the best model for given problem. 


* Intensive Care Units (ICUs) are continually faced with the difficulty of monitoring patients for the development of sepsis (an infection that can accrue while staying in ICU).

* This is the motivation for developing the machine learning model. There are two main purposes: 

1. Reducing the risk of health complications.
2. Manage the ICU resources (such as bed availability, etc.).

* The training dataset provides list of essential attributes (features) related to: patient characteristics, diagnoses, treatments, services, hospital charges and patients socio-economic background.



# Ultimate Judgement and Analysis
So far `Logistic Regression` is the best approach among those models and methods to use in the real world with our problem based on below criteria: 
* Logistic Regression is not overfitting easily compared to other model. 

* It provides the smallest generalization GAP. With F1 score of 69% in testing dataset and 70% in the training dataset. 

* Accruracy is the highest with 76% on testing set, with the smallest different compared to training accuracy of 77.232%. 

* False Negative is lowest 8.67%. 

* The accuracy when classified the patient in class 1 is the second highest (0.64)

* The ability to detect a class 0 and 1 is nearest to each other, and is considerably high of 75%. 

***Machine Learning model issue discussion:***

From the table below we can say that: 

<img src="./Model_Comparison.png" alt="Table compares all models" style="height: 280px; width:850px;"/>

* First I will discuss about the overfitting problem, Decision Tree without Hyperparameter Tuning and Random Forest seems to be overfitted with 100% of accuracy in the training dataset, but perform poorly in the testing dataset around 50%. The other models still have a considerable GAP between the accuracy or F1 score in the testing and training dataset  > 5%, which is Decision Tree with hyperparameter tuning and base model of logistic regression (without regulization). However, in models with resampling techniques applied in the pipeline with Logistic Regression -> the model seems not to be overfitted. With the application of regularisation in Logistic Regression, the model is prevented from being overfitting as we put penalty in the parameter. This is such an important things to note, as in the base model which is without regularisation we saw a huge generalisation GAP, and the significant higher accuracy on the training dataset. 

* According to the dataset, the accuracy of Logistic Regression with Regularisation + SMOTE technique is the highest among those 4 models which is ***76.7%***. As already mentioned before, the accuracy score is optimal evaluation metrics for this issue, and also this metric is not wisely used to evaluate the imbalanced dataset, however, this also needed to be taken into account in our case. 
The most important metric we want to notice is F1 score. Among all the models, the Logistic Regression with Regularisation + SMOTE technique have the highest score - it is stable in the cross validation process with 0.6908 as a mean for 5-folds.  This model is the one that have most generalized characteristic as the GAP of F1 between train and test sets is quite small 0.98. 

* According to confusion matrix report of each model, Logisitic Regression with regularisation and SMOTE Technique have the best False Negative of 8.47% (patient who is likely to develop sepssis but is predicted not to), and considerated low False Positive of 14.67%. The model is also good at predicting who is not likely to develop a sepssis with nearly 0.8 percentage of accuracy in recall. It can be able to determine how many patients at class 0 being classified correctly. While precision of class 0 can also tells us that model is good at  differ one class 0  from all others. 
* Taking context of the problem to this model, we care about 2 factors mitigate the risk and manage the resource of ICU well. In the first purpose, the chosen model is pretty well achieved. As the dataset is imbalanced and many cases is reported not to develop a sepssis during their stay at ICU. The model do it greatest job with 70% to detect the patient who is likely to develop sepssis. The noticeable thing here is the model have the lowest rate of False Negative. Since we more care about the positive detecting, we want to “Predict if a given in ICU would not develop a sepsis (Sepsis Negative / class 0) or will develop sepsis (Sepsis Positive / class 1) during their ICU stay”. And Sepssis is a life-threatening infection, we wants to minimize the rate of False Negative in our model, with good F1 score. In this case, the F1 score of 0.7 or 70% is acceptable.
* In term of resource management for ICU, the model can be able to detect and predict class 0 accurately with presion and recall of 0.85 and 0.78, respectively -> F1 score is 0.81 in the class 0. But at the same time, it is still can detect and predict the patient with class 1 at a moderated rate. Although, the False Positive rate is not the lowest in all above confusion matrix, it is a good rate of 14.67%. This won't put much pressure on the resource since we only incorrectly patient is is not likely to develop sepssis to have sepssis, but not much so the resource spend to check up on them is not much. 

* Limitation of chosen model: 
- It establishes linear boundaries. Logistic regression requires a linear relationship between the independent variables and the log odds.
- The primary constraint for Logistic Regression is the linear relationship between the dependent and independent variables.
- Algorithms that are more powerful and compact, such as Neural Networks, can easily surpass this one.




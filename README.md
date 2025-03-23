# Project_4
The project goal is to develop a Random Forest Classifier ML model to classify the bleaching areas of coral in Fiji
- There are 5 classes with class 0 is area with no bleaching coral and 5 is area with heavily bleaching coral
- The model is first biased toward the majority class (class 0) with high accuracy of 91%
- Then, the model data is resampled with SMOTE and hyperparamters tuning with RandomizedSearchCV to be less bias with lower accuracy of 89%s
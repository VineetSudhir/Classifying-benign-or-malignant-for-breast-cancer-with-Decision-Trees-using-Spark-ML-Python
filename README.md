# Classifying-benign-or-malignant-for-breast-cancer-with-Decision-Trees-using-Spark-ML
ML Pipeline, Visualization, Hyperparameter Tuning, and with Persist

Feature Transformation
  Convert features for each row into a vector of numbers creating a matrix for the entire dataset (all rows). Matrix representation of all training data enables much     more efficient implementation of ML algorithms.
  Utilized StringIndexer function to convert string feature and class label into double.
  Utilized VectorAssembler function to convert selected features into a vector of numbers.
  Then Decision Tree Learning is conducted.
  Convert indexed predictions to the class label using IndexToString function to compare and calculate F1-score.
  Combined all these steps into a pipeline.


Highest F1 score: 0.9833751140402666 with max depth of 4 and 2 minimum instances per node

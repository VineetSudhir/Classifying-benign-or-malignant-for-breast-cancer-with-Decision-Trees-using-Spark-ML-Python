import pyspark
import pandas as pd
import csv
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from decision_tree_plot.decision_tree_parser import decision_tree_parse
from decision_tree_plot.decision_tree_plot import plot_trees


ss=SparkSession.builder.appName("DT").getOrCreate()


data = ss.read.csv("./breast-cancer-wisconsin.data.txt", header=True, inferSchema=True)

data.printSchema()

data.show(5)

from pyspark.sql.functions import col
class_count = data.groupBy(col("class")).count()
class_count.show()

bnIndexer = StringIndexer(inputCol="bare_nuclei", outputCol="bare_nuclei_index").fit(data)
bnIndexer

transformed_data = bnIndexer.transform(data)
transformed_data.show(4)


labelIndexer= StringIndexer(inputCol="class", outputCol="indexedLabel").fit(data)
labelIndexer

transformed2_data = labelIndexer.transform(transformed_data)
transformed2_data.show(4)


input_features = ['clump_thickness', 'unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei_index', 'bland_chrom', 'norm_nucleoli', 'mitoses']

assembler = VectorAssembler(inputCols=input_features, outputCol="features")
assembler

transformed3_data = assembler.transform(transformed2_data)
transformed3_data.show(4)

selected_transformed3_data = transformed3_data.select("features",'indexedLabel')
selected_transformed3_data.show(5)



trainingData3, testData3= transformed3_data.randomSplit([0.75, 0.25], seed=1234)


dt=DecisionTreeClassifier(featuresCol="features", labelCol="indexedLabel", maxDepth=6, minInstancesPerNode=2)
dt

dt_model = dt.fit(trainingData3)


dt_model


test_prediction = dt_model.transform(testData3)
test_prediction.persist().show(3)


test_prediction.select("features","class","indexedLabel", "rawPrediction", "probability", "prediction").show(5)

labelIndexer.labels

labelConverter=IndexToString(inputCol="prediction", outputCol="predictedClass", labels=labelIndexer.labels)

test2_prediction = labelConverter.transform(test_prediction)


test2_prediction.select("features","class","indexedLabel","prediction","predictedClass").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")


f1 = evaluator.evaluate(test_prediction)
print("f1 score:", f1)
# f1 score: 0.9780273904005113




trainingData, testData= data.randomSplit([0.75, 0.25], seed=1234)


assembler = VectorAssembler( inputCols=input_features, outputCol="features")
dt=DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", maxDepth=5, minInstancesPerNode=2)
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedClass", labels=labelIndexer.labels)
pipeline = Pipeline(stages=[labelIndexer, bnIndexer, assembler, dt, labelConverter])
model = pipeline.fit(trainingData)
test_predictions = model.transform(testData)


pipeline

model

test_predictions.select("class","indexedLabel","prediction","predictedClass").show(10)


evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")


f1 = evaluator.evaluate(test_predictions)
print("f1 score of testing data:", f1)
# f1 score of testing data: 0.9726090922212769


# Decision Tree Visualization

DTmodel = model.stages[3]
print(DTmodel)


model_path="./DTmodel_vis"

tree=decision_tree_parse(DTmodel, ss, model_path)
column = dict([(str(idx), i) for idx, i in enumerate(input_features)])
plot_trees(tree, column = column, output_path = '/storage/home/vas5260/Lab7DT/DTtree2.html')


trainingData, testingData= data.randomSplit([0.75, 0.25], seed=1234)
model_path="./DTmodel_vis"
trainingData.persist()
testingData.persist()


hyperparams_eval_df = pd.DataFrame( columns = ['max_depth', 'minInstancesPerNode', 'training f1', 'testing f1', 'Best Model'] )
index =0 
highest_testing_f1 = 0
max_depth_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
minInstancesPerNode_list = [2, 3, 4, 5, 6]
assembler = VectorAssembler( inputCols=input_features, outputCol="features")
labelConverter = IndexToString(inputCol = "prediction", outputCol="predictedClass", labels=labelIndexer.labels)
for max_depth in max_depth_list:
    for minInsPN in minInstancesPerNode_list:
        seed = 37
        dt= DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", maxDepth= max_depth, minInstancesPerNode=minInsPN)
        pipeline = Pipeline(stages=[labelIndexer, bnIndexer, assembler, dt, labelConverter])
        model = pipeline.fit(trainingData)
        training_predictions = model.transform(trainingData)
        testing_predictions = model.transform(testingData)
        evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="f1")
        training_f1 = evaluator.evaluate(training_predictions)
        testing_f1 = evaluator.evaluate(testing_predictions)
        hyperparams_eval_df.loc[index] = [ max_depth, minInsPN, training_f1, testing_f1, 0]  
        index = index +1
        if testing_f1 > highest_testing_f1 :
            best_max_depth = max_depth
            best_minInsPN = minInsPN
            best_index = index -1
            best_parameters_training_f1 = training_f1
            best_DTmodel= model.stages[3]
            best_tree = decision_tree_parse(best_DTmodel, ss, model_path)
            column = dict( [ (str(idx), i) for idx, i in enumerate(input_features) ])           
            highest_testing_f1 = testing_f1
print('The best max_depth is ', best_max_depth, ', best minInstancesPerNode = ',       best_minInsPN, ', testing f1 = ', highest_testing_f1) 
plot_trees(best_tree, column = column, output_path = '/storage/home/vas5260/Lab7DT/bestDTtree2B.html')



hyperparams_eval_df.loc[best_index]=[best_max_depth, best_minInsPN, best_parameters_training_f1, highest_testing_f1, 1000]


schema3= StructType([ StructField("Max Depth", FloatType(), True), StructField("MinInstancesPerNode", FloatType(), True ), StructField("Training f1", FloatType(), True), StructField("Testing f1", FloatType(), True), StructField("Best Model", FloatType(), True)])



HyperParams_Tuning_DF = ss.createDataFrame(hyperparams_eval_df, schema3)



output_path = "./DT_output"
HyperParams_Tuning_DF.rdd.saveAsTextFile(output_path)


ss.stop()






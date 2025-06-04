# # # run spark spark-submit *.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import lower
from pyspark.ml.feature import Word2Vec, RegexTokenizer, StringIndexer, Word2VecModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import os
import time


# Start Spark session
# spark = SparkSession.builder.appName("word2vecML").getOrCreate()
spark = (
    SparkSession.builder
    .appName("word2vecML")
    # .master(f"local[1]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

def load_data(file, columns, label, diagnostic):
    Dataset = pd.read_csv(file, usecols=columns + [label])
    Dataset = Dataset.fillna('')
    Dataset[diagnostic] = Dataset[columns[0]]
    return Dataset[[diagnostic, label]] # return a dataset that will only use left and right keyword diagnostics



def lr (features, labels, iter, train):

    Logistic = LogisticRegression(featuresCol=features, labelCol=labels, regParam=0.3, elasticNetParam=0.1)
    Logistic.setMaxIter(iter)
    model = Logistic.fit(train)
    return model


def model(filepath,columns, label, diagnostic,iter, train, test):

    df = load_data(filepath,columns=columns,label=label, diagnostic=diagnostic) # only print either left/right diagnostic keyword and include the label
    # print(df[:5])
    df1 = spark.createDataFrame(df)



    #  Lowercase the diagnostic column to avoid case mismatch make sure that Retina = retina
    df1_cleaned = df1.withColumn("diagnostic_lower", lower(df1[diagnostic]))

    #  split on non-word characters like spaces, commas, etc. Retina, normal = > Retina normal
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.RegexTokenizer.html referenced in how to use RegexTokenizer
    tokenizer = RegexTokenizer(inputCol="diagnostic_lower", outputCol="words", pattern="\\W")
    word_tokens = tokenizer.transform(df1_cleaned)

    # reference for word2vec in spark https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Word2Vec.html
    word2vec = Word2Vec(vectorSize=20, inputCol="words", outputCol="word2vec_Output_vector",seed=42)
    word2vec.setMaxIter(100)
    model_word2vec = word2vec.fit(word_tokens)
    result_word2vec = model_word2vec.transform(word_tokens)

    index_labels = StringIndexer(inputCol=label,outputCol="label_index").setStringOrderType("alphabetAsc")  # maps a string column of labels to an ML column of labels in Alphabetical order

    df_index = index_labels.fit(result_word2vec).transform(result_word2vec)  # final data frame to pass on model


    training_set, test_set = df_index.randomSplit([train, test]) #split training and test

    labels_lr = "label_index"

    model_lr = lr(features="word2vec_Output_vector", labels=labels_lr, iter=iter, train=training_set)

    return model_lr, labels_lr,test_set,result_word2vec,model_word2vec


def save(path_name:str,name: str,model):

    path = os.path.join(path_name,name)
    model.save(path) # save model to corresponding path folder and name




if __name__ =="__main__":
    # Predict on test
    start_time = time.time()
    Model1,labels,test,result_word2vec,model_word2vec =model(filepath= "left_eye.csv",
                                             columns=['Left-Diagnostic Keywords'],
                                             diagnostic="Diagnostic_patient_left",
                                             iter=200,
                                             train=0.8,
                                             test=0.2,
                                             label="labels"
                                             )

    Model2, labels1, test1, result_word2vec1, model_word2vec1 = model(filepath="right_eye.csv",
                                                                 columns=['Right-Diagnostic Keywords'],
                                                                 diagnostic="Diagnostic_patient_right",
                                                                 iter=200,
                                                                 train=0.8,
                                                                 test=0.2,
                                                                 label="labels"
                                                                 )




    save(path_name="model_trained",name="word2vec_model_left_eye",model=model_word2vec) # saves the word2vec model left eye dataset
    save(path_name="model_trained",name = "lr_model_left_eye",model=Model1)

    save(path_name="model_trained", name="word2vec_model_right_eye",model=model_word2vec1)  # saves the word2vec model right eye dataset
    save(path_name="model_trained", name="lr_model_right_eye", model=Model2)


    predictionsM1 = Model1.transform(test) # get the predictions from test set
    predictionsM2 = Model2.transform(test1)

    # Evaluate
    evaluator_m1 = MulticlassClassificationEvaluator(labelCol=labels, predictionCol="prediction",metricName="accuracy") # evaluate using multiclassification
    accuracy_m1 = evaluator_m1.evaluate(predictionsM1)

    evaluator_m2 = MulticlassClassificationEvaluator(labelCol=labels1, predictionCol="prediction",metricName="accuracy")  # evaluate using multiclassification
    accuracy_m2 = evaluator_m1.evaluate(predictionsM2)

    print(f"Classification Accuracy on Model 1 with left eye: {accuracy_m1}")
    # # right eye csv ~ 82% accuracry on test set with train = 0.8, test = 0.2
    # # left eye csv ~ 83% accuracy on test set

    print(f"Classification Accuracy on Model 2 with right eye: {accuracy_m2}")

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for running left and right model: {elapsed_time:.4f} seconds")

    spark.stop()
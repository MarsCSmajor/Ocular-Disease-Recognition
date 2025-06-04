from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2VecModel, RegexTokenizer
from pyspark.sql.functions import lower
import pandas as pd
# DONE BY MICHAEL
def vectorize_eye(csv_path, model_path, keyword_column, output_csv):
    print(f"Processing {csv_path} with model at {model_path}")

    # Spark
    spark = SparkSession.builder.appName("IamGabriel").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")



    df_pd = pd.read_csv(csv_path).fillna('')
    df_spark = spark.createDataFrame(df_pd)



    df_clean = df_spark.withColumn("diagnostic_lower", lower(df_spark[keyword_column]))
    tokenizer = RegexTokenizer(inputCol="diagnostic_lower", outputCol="words", pattern="\\W")
    tokenized = tokenizer.transform(df_clean)



    # Load model and transform
    w2v_model = Word2VecModel.load(model_path)
    vectorized = w2v_model.transform(tokenized)



    final_df = vectorized.drop("diagnostic_lower", "words") \
                     .drop(keyword_column) \
                     .withColumnRenamed("word2vec_Output_vector", keyword_column)


    # Save
    final_df.toPandas().to_csv(output_csv, index=False)
    spark.stop()


if __name__ == "__main__":
    vectorize_eye(
        csv_path="left_eye.csv",
        model_path="model_trained/word2vec_model_left_eye",
        keyword_column="Left-Diagnostic Keywords",
        output_csv="left_eye_new.csv"
    )

    vectorize_eye(
        csv_path="right_eye.csv",
        model_path="model_trained/word2vec_model_right_eye",
        keyword_column="Right-Diagnostic Keywords",
        output_csv="right_eye_new.csv"
    )

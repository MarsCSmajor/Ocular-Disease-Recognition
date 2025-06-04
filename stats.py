from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, trim, initcap, lower, avg, sum as _sum, round as spark_round
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import time

mysql_jar_path = "/home/cs179g/model_training/mysql-connector-java-8.0.30/mysql-connector-java-8.0.30.jar"

connection_properties = {
        "user": "teammate",
        "password": "2025",
        "driver": "com.mysql.cj.jdbc.Driver"
}

jdbc_url = "jdbc:mysql://localhost:3306/tabular_data_stats"


def write_to_mysql(df, table_name):
    df.write.jdbc(
            url = jdbc_url,
            table = table_name,
            mode = "overwrite",
            properties = connection_properties
            )

# Initializing Spark session
spark = SparkSession.builder.appName("Ocular Disease Stats").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load data
data_path_full = "/home/cs179g/OcularDiseaseDataSet/full_df.csv"
df = spark.read.format("csv").option("header", "true").load(data_path_full)

# Cast age to integer
df = df.withColumn("Patient Age", col("Patient Age").cast("int"))

diagnosis_attr = ["N","D", "G", "C", "A", "H", "M", "O"]
for attr_label in diagnosis_attr:
    df = df.withColumn(attr_label,col(attr_label).cast("int"))

# Add Age Group column
df_age_group = df.withColumn("Age Group",
                    when(col("Patient Age").between(0, 20), "0-20")
                     .when(col("Patient Age").between(21, 30), "21-30")
                    .when(col("Patient Age").between(31, 40), "31-40")
                    .when(col("Patient Age").between(41, 50), "41-50")
                    .when(col("Patient Age").between(51, 60), "51-60")
                    .when(col("Patient Age").between(61,70), "61-70")
                    .when(col("Patient Age").between(71,80), "71-80")
                    .otherwise("81+"))
#Total number of distinct participants
print("Total number of distinct participants based on ID:", df.select("ID").distinct().count())

#Total number of ALL participants
print("Total number of participants in dataset:", df.select("ID").count())

# Count of patients per age group
print("Count of ALL patients per age group:")
df_age_group.groupBy("Age Group").count().orderBy("Age Group").show()

print("Count of Distinct Patients Per Age Group")
df_age_group.select("ID","Age Group")\
        .dropDuplicates( ["ID"] ) \
        .groupBy("Age Group") \
        .count() \
        .orderBy("Age Group")\
        .show()

# Count of patients based on sex
print("Count of patients based on sex:")
df.groupBy("Patient Sex").count().orderBy("Patient Sex").show()

print("Count of DISTINCT patients by sex:")
distinct_by_sex = df.select("ID", "Patient Sex").dropDuplicates(["ID"]).groupBy("Patient Sex").count()
distinct_by_sex.show()
write_to_mysql(distinct_by_sex, "distinct_patients_by_sex")

#Total patient count per diagnosis
print("Total patients per diagnosis:")
df.select([_sum(col(d)).alias(f"Total_{d}") for d in diagnosis_attr]).show()

print("Diagnosis frequency by gender (distinct patients):")
distinct_diagnosis_sex = df.dropDuplicates(["ID"]).groupBy("Patient Sex")\
            .agg(*[_sum(col(d)).alias(f"Total_{d}") for d in diagnosis_attr])
distinct_diagnosis_sex.show()
write_to_mysql(distinct_diagnosis_sex, "distinct_diagnosis_by_sex")

#Per Diagnosis, number of patient per age group
print("Diagnosis frequency by age group:")
df_age_group.groupBy("Age Group") \
            .agg(*[_sum(col(d)).alias(f"Total_{d}") for d in diagnosis_attr]) \
            .orderBy("Age Group") \
            .show()
df_diagnosis_by_age = df_age_group.groupBy("Age Group") \
        .agg(*[_sum(col(d)).alias(f"Total_{d}") for d in diagnosis_attr])
df_diagnosis_by_age.orderBy("Age Group").show()

write_to_mysql(df_diagnosis_by_age, "diagnosis_by_age_group")

print("Diagnosis frequency by age group (distinct patients):")
distinct_diag_age = df_age_group.dropDuplicates(["ID"]).groupBy("Age Group")\
            .agg(*[_sum(col(d)).alias(f"Total_{d}") for d in diagnosis_attr])
distinct_diag_age.show()
write_to_mysql(distinct_diag_age, "distinct_diagnosis_by_age_group")



#Average age per diagnosis
print("Average age per diagnosis:")
for d in diagnosis_attr:
    df.filter(col(d) == 1).agg(spark_round(avg("Patient Age")).alias(f"Avg Age for {d}")).show()

print("Count of patients with more than on diagnosis")
df_age_group = df_age_group.withColumn("Diagnosis Count", col("N") + col("D") + col("G") + col("C") + col("A") + col("H")+ col("M")+ col("O"))
multipleDiagcount = df_age_group.filter(col("Diagnosis Count") > 1).count()
print(multipleDiagcount)

#Demographic info. of patients with more than 1 diagnosis
print("Demographics of patients with multiple diagnoses:")
df_age_group.filter(col("Diagnosis Count") > 1)\
            .groupBy("Patient Sex","Age Group")\
            .count()\
            .orderBy("Age Group", "Patient Sex")\
            .show()

#Analysis of Variance: Does average number of diagnosis differ across age groups
print("Average number of diagnosis across age groups")
df_age_group.groupBy("Age Group") \
        .agg(
                spark_round(avg("Diagnosis Count"), 2).alias("Avg Diagnosis Count"),
                count("ID").alias("Total Patients"
        )) \
        .orderBy("Age Group") \
        .show()
df_avg_diag_by_age = df_age_group.groupBy("Age Group") \
            .agg(
                    spark_round(avg("Diagnosis Count"), 2).alias("Avg Diagnosis Count"),
                    count("ID").alias("Total Patients")
            )

plot_data = df_avg_diag_by_age.orderBy("Age Group").collect()
x = [row["Age Group"] for row in plot_data]
y = [row["Avg Diagnosis Count"] for row in plot_data]

plt.figure(figsize=(10, 6))
plt.bar(x, y)
plt.xlabel("Age Group")
plt.ylabel("Average Diagnosis Count")
plt.title("Average Diagnoses per Patient by Age Group")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plot_path = "/home/cs179g/avg_diag_by_age.png"
plt.savefig(plot_path)

#Logistic Regression to Predict the Likelihood of having a specific diagnosis based on age & sex
for label in diagnosis_attr:
    print(f"\n=== Logistic Regression for Diagnosis: {label} ===")

    df_lr_model = df.select("Patient Age", "Patient Sex", label)
    gender_indexer = StringIndexer(inputCol="Patient Sex", outputCol="SexIndexed", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=["Patient Age", "SexIndexed"], outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol=label, maxIter=10)
    pipeline = Pipeline(stages=[gender_indexer, assembler, lr])

    model = pipeline.fit(df_lr_model)
    predictions = model.transform(df_lr_model)

    evaluator = BinaryClassificationEvaluator(labelCol=label)
    auc = evaluator.evaluate(predictions)

    lr_model = model.stages[-1]
    print(f"AUC: {auc:.4f}")
    print(f"Intercept: {lr_model.intercept}")
    print(f"Coefficients [Age, Sex]: {lr_model.coefficients}")

#Execution time graph with varying data size
limits = [1000, 10000, 50000, 100000]
time_results = []

for limit in limits:
    print(f"\nRunning {limit} rows")

    df_subset = df_age_group.limit(limit)

    start_time = time.time()

    df_subset.groupBy("Age Group") \
            .agg(count("ID").alias("Patient Count")) \
            .orderBy("Age Group") \
            .collect()
    end_time = time.time()
    duration = __builtins__.round(end_time - start_time, 2)

    print(f"Execution tims: {duration} seconds")
    time_results.append( (limit,duration) )

print("\n=== Summary Table: Execution Time by Data Size ===")
print("Rows\tTime (s)")
for row_count, timing in time_results:
        print(f"{row_count}\t{timing}")


# Read a table from MySQL
table_to_read = "diagnosis_by_age_group"
df_read_back = spark.read.jdbc(
    url=jdbc_url,
    table=table_to_read,
    properties=connection_properties
)

print(f"\n=== Contents of Table: {table_to_read} ===")
df_read_back.show()

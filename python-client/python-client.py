from minio import Minio
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from delta import configure_spark_with_delta_pip
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.spark
from mlflow.models.signature import infer_signature
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

builder = SparkSession.builder.appName("LakehouseMLExample") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.metrics.impl", "org.apache.hadoop.fs.s3a.SimpleMetricsContext") \
    .config("spark.hadoop.fs.s3a.metrics.reporting.enabled", "false")

spark = configure_spark_with_delta_pip(builder, extra_packages=["org.apache.hadoop:hadoop-aws:3.3.4"]).getOrCreate()

# 
logger.info("Spark session created successfully.")

client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

minio_bucket = "delta-lake"

found = client.bucket_exists(minio_bucket)
if not found:
    client.make_bucket(minio_bucket)
# 
bucket_name = "mlflow"
found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)

logger.info("MinIO Client Connected!")

os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
os.environ["MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING"] = "true"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
# 
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("My ML Experiment")

logger.info("Starting to read CSV file in spark...")
data_csv = './data/uci-secom.csv'
data_df = spark.read.format('csv').option('header', 'true').option('inferSchema', 'true').load(data_csv)

logger.info("Finish read CSV.")

data_df.write.format("delta").mode("overwrite").save(f"s3a://{minio_bucket}/uci-secom")

logger.info("Finish writing to delta lake.")

delta_path = f's3a://{minio_bucket}/uci-secom'

delta_df = spark.read.format("delta").load(delta_path)
delta_df = delta_df.withColumn("Time", col("Time").cast("string"))
pandas_df = delta_df.toPandas()
# 
logger.info("Finish convert to pandas data frame.")

X = pandas_df.drop(["Pass/Fail", "Time"], axis=1)
y = pandas_df["Pass/Fail"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
logger.info("Finish split data for ML model training.")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 
    logger.info("Finish train a Random Forest model.")

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy}")

    signature = infer_signature(X_train, model.predict(X_train))

    input_example = X_train.iloc[:1]
    # 
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature, input_example=input_example)

logger.info("MLflow Tracking Complete!")

spark.stop()

logger.info("Spark session stopped.")
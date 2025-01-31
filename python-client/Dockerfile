FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y openjdk-11-jdk curl unzip && apt-get clean

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV SPARK_VERSION=3.3.0
ENV HADOOP_VERSION=3
ENV DELTA_CORE_VERSION=2.2.0
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV MINIO_ENDPOINT=http://minio:9000
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /opt
RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ENV PATH="$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"

WORKDIR /app

COPY python-client.py .

CMD ["python", "python-client.py"]
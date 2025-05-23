# Databricks notebook source
# MAGIC %md
# MAGIC ##Widgets

# COMMAND ----------

dbutils.widgets.text("min_quality", "0", "Minimum Quality Score")
min_quality_widget = int(dbutils.widgets.get("min_quality"))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, max, count, lag, datediff, struct
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType
import xml.etree.ElementTree as ET
from pyspark.sql.functions import udf
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC #Create a SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("RadiologyTimeline").getOrCreate()

# COMMAND ----------

radiology_images_data = [
    (1, 'CT', '2025-01-01', 'study1', 90, "<Report><Title>CT Head</Title><Impression>No acute findings.</Impression></Report>"),
    (1, 'MRI', '2025-01-15', 'study2', 95, "<Report><Title>MRI Brain</Title><Impression>Stable chronic changes.</Impression></Report>"),
    (2, 'XRAY', '2024-12-20', 'study3', 85, "<Report><Title>Chest X-Ray</Title><Impression>Clear lungs.</Impression></Report>"),
    (1, 'CT', '2025-02-10', 'study4', 92, "<Report><Title>CT Abdomen</Title><Impression>Normal study.</Impression></Report>"),
    (3, 'MRI', '2025-03-01', 'study5', 88, "<Report><Title>MRI Spine</Title><Impression>Mild disc bulge.</Impression></Report>"),
    (2, 'CT', '2025-03-15', 'study6', 91, "<Report><Title>CT Chest</Title><Impression>No significant abnormality.</Impression></Report>"),
    (1, 'XRAY', '2025-03-20', 'study7', 78, "<Report><Title>X-Ray Knee</Title><Impression>Mild osteoarthritis.</Impression></Report>"),
    (3, 'CT', '2025-04-01', 'study8', 96, "<Report><Title>CT Head</Title><Impression>No change.</Impression></Report>"),
]

# COMMAND ----------

radiology_images_schema = StructType([
    StructField("PatientID", IntegerType(), False),
    StructField("Modality", StringType(), False),
    StructField("StudyDate", StringType(), False),
    StructField("StudyID", StringType(), False),
    StructField("QualityScore", IntegerType(), False),
    StructField("ReportXML", StringType(), False),
])

# COMMAND ----------

radiology_images_df = spark.createDataFrame(radiology_images_data, radiology_images_schema)

# COMMAND ----------

radiology_images_df = radiology_images_df.withColumn("StudyDate", col("StudyDate").cast(DateType()))

# COMMAND ----------

patients_data = [
    (1, 'John Doe', '1980-05-10', 'Male'),
    (2, 'Jane Smith', '1992-11-22', 'Female'),
    (3, 'Peter Jones', '1975-03-01', 'Male'),
]

# COMMAND ----------

patients_schema = StructType([
    StructField("PatientID", IntegerType(), False),
    StructField("Name", StringType(), False),
    StructField("DOB", StringType(), False),
    StructField("Gender", StringType(), False),
])

# COMMAND ----------

patients_df = spark.createDataFrame(patients_data, patients_schema)
patients_df = patients_df.withColumn("DOB", col("DOB").cast(DateType()))

# COMMAND ----------

# MAGIC %md
# MAGIC Step 1: Filter Images based on QualityScore

# COMMAND ----------

filtered_images_df = radiology_images_df.filter(col("QualityScore") >= min_quality_widget)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2: Parse XML Reports

# COMMAND ----------

def parse_report_xml(xml_string):
    try:
        root = ET.fromstring(xml_string)
        title = root.findtext('Title')
        impression = root.findtext('Impression')
        return (title, impression)
    except Exception as e:
        return (None, None)

# COMMAND ----------

parse_report_udf = udf(parse_report_xml, StructType([
    StructField("ReportTitle", StringType()),
    StructField("Impression", StringType())
]))

# COMMAND ----------

parsed_reports_df = filtered_images_df.withColumn("ReportDetails", parse_report_udf(col("ReportXML"))) \
    .withColumn("ReportTitle", col("ReportDetails").getItem("ReportTitle")) \
    .withColumn("Impression", col("ReportDetails").getItem("Impression")) \
    .drop("ReportXML", "ReportDetails")

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3: Calculate Time Since Last Study

# COMMAND ----------

window_spec = Window.partitionBy("PatientID").orderBy("StudyDate")

# COMMAND ----------

time_features_df = parsed_reports_df.withColumn(
    "TimeSinceLast",
    datediff(col("StudyDate"), lag(col("StudyDate"), 1).over(window_spec))
)

# COMMAND ----------

time_features_df = parsed_reports_df.withColumn(
    "TimeSinceLast",
    datediff(col("StudyDate"), lag(col("StudyDate"), 1).over(window_spec))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 4: Pivot Modality to Get Last Study Dates and Count Total Studies

# COMMAND ----------

pivoted_df = time_features_df.groupBy("PatientID") \
    .agg(
        max(when(col("Modality") == "CT", col("StudyDate"))).alias("Last_CT"),
        max(when(col("Modality") == "MRI", col("StudyDate"))).alias("Last_MRI"),
        max(when(col("Modality") == "XRAY", col("StudyDate"))).alias("Last_XRAY"),
        count("*").alias("Total_Studies")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Step 5: Join with Patients DataFrame

# COMMAND ----------

final_df = pivoted_df.join(patients_df, "PatientID", "inner") \
    .select("PatientID", "Name", "DOB", "Gender", "Last_CT", "Last_MRI", "Last_XRAY", "Total_Studies")

# COMMAND ----------


# Display the final DataFrame
final_df.display()
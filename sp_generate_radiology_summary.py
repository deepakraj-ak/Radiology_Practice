# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    when,
    count,
    avg,
    sum,
    datediff,
    current_timestamp,
    date_sub,
    to_date,
    expr
)

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    TimestampType,
    BooleanType,
    FloatType,
    DateType
)

# COMMAND ----------

from datetime import datetime, timedelta
import uuid

# COMMAND ----------

# MAGIC %md
# MAGIC  PARAMETERS AND METADATA

# COMMAND ----------

dbutils.widgets.text("days_back", "7", "Number of Days Back for Reports")
days_back_widget = int(dbutils.widgets.get("days_back"))

# COMMAND ----------

end_date = datetime.now()
start_date = end_date - timedelta(days=days_back_widget)
today_timestamp = datetime.now()
job_id = str(uuid.uuid4())

# COMMAND ----------

print(f"Start: {start_date}, End: {end_date}")

# COMMAND ----------

job_name = "sp_generate_radiology_summary_pyspark"

# COMMAND ----------



# COMMAND ----------

spark = SparkSession.builder.appName("RadiologySummaryGenerator").getOrCreate()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Sample data

# COMMAND ----------

radiology_reports_data = [
    (1, 101, datetime(2025, 5, 20, 10, 0, 0), True, "Report text 1"),
    (2, 102, datetime(2025, 5, 21, 11, 0, 0), False, "Report text 2"),
    (3, 103, datetime(2025, 5, 22, 12, 0, 0), True, "Report text 3"),
    (4, 104, datetime(2025, 5, 15, 9, 0, 0), False, "Report text 4 (old)"),
    (5, 105, datetime(2025, 5, 23, 14, 0, 0), False, "Report text 5"),
    (6, 106, datetime(2025, 5, 23, 15, 0, 0), True, "Report text 6"),
    (7, 107, datetime(2025, 5, 23, 16, 0, 0), False, "Report text 7"),
    (8, 108, datetime(2025, 5, 23, 17, 0, 0), True, "Report text 8"),
    (9, 109, datetime(2025, 5, 23, 18, 0, 0), False, "Report text 9"),
    (10, 110, datetime(2025, 5, 23, 19, 0, 0), True, "Report text 10"),
]

# COMMAND ----------

radiology_reports_schema = StructType([
    StructField("report_id", IntegerType(), False),
    StructField("order_id", IntegerType(), False),
    StructField("report_datetime", TimestampType(), False),
    StructField("critical_finding_flag", BooleanType(), False),
    StructField("report_text", StringType(), False),
])

# COMMAND ----------

radiology_reports_df = spark.createDataFrame(radiology_reports_data, radiology_reports_schema)

# COMMAND ----------

radiology_orders_data = [
    (101, 1, 'CT', datetime(2025, 5, 20, 9, 30, 0), 1001, 201),
    (102, 2, 'MRI', datetime(2025, 5, 21, 10, 30, 0), 1002, 202),
    (103, 3, 'XRAY', datetime(2025, 5, 22, 11, 30, 0), 1001, 201),
    (104, 4, 'XR', datetime(2025, 5, 15, 8, 30, 0), 1003, 203),
    (105, 5, 'CT', datetime(2025, 5, 23, 13, 30, 0), 1002, 202),
    (106, 6, 'US', datetime(2025, 5, 23, 14, 30, 0), 1001, 201),
    (107, 7, 'ULTRASOUND', datetime(2025, 5, 23, 15, 30, 0), 1003, 203),
    (108, 8, 'MRI', datetime(2025, 5, 23, 16, 30, 0), 1002, 202),
    (109, 9, 'MR', datetime(2025, 5, 23, 17, 30, 0), 1001, 201),
    (110, 10, 'CT', datetime(2025, 5, 23, 18, 30, 0), 1003, 203),
    (111, 11, 'CT', None, 1001, 201), # Missing order_datetime for DQ check
    (112, 12, 'XRAY', datetime(2025, 5, 23, 10, 0, 0), 1002, 202), # Report not in recent reports
]

# COMMAND ----------

radiology_orders_schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("patient_id", IntegerType(), False),
    StructField("modality", StringType(), False),
    StructField("order_datetime", TimestampType(), True), # Allow null for DQ check
    StructField("location_id", IntegerType(), False),
    StructField("attending_physician_id", IntegerType(), False),
])

# COMMAND ----------

radiology_orders_df = spark.createDataFrame(radiology_orders_data, radiology_orders_schema)

# COMMAND ----------

physicians_data = [
    (201, 'Dr. Praveen Raj'),
    (202, 'Dr. Preethi Badam'),
    (203, 'Dr.  Sushmitha Kannan'),
    (204, 'Dr. Perarasu'), # Physician not in orders
]

# COMMAND ----------

physicians_schema = StructType([
    StructField("physician_id", IntegerType(), False),
    StructField("name", StringType(), False),
])

# COMMAND ----------

physicians_df = spark.createDataFrame(physicians_data, physicians_schema)

# COMMAND ----------

locations_data = [
    (1001, 'PSG Hospital'),
    (1002, 'KMCH Hospital'),
    (1003, 'Kumaran Hospital'),
    (1004, 'KG Hospitals'), # Location not in orders
]

# COMMAND ----------

locations_schema = StructType([
    StructField("location_id", IntegerType(), False),
    StructField("location_name", StringType(), False),
])

# COMMAND ----------

locations_df = spark.createDataFrame(locations_data, locations_schema)


# COMMAND ----------

# MAGIC %md
# MAGIC STEP 0: LOG START

# COMMAND ----------

print(f"[{today_timestamp}] Job '{job_name}' (ID: {job_id}) STARTED.")
audit_log_data = [(job_id, job_name, today_timestamp, None, 'STARTED', None)]
audit_log_schema = StructType([
    StructField("job_id", StringType(), False),
    StructField("job_name", StringType(), False),
    StructField("start_time", TimestampType(), False),
    StructField("end_time", TimestampType(), True),
    StructField("status", StringType(), False),
    StructField("error_message", StringType(), True),
])

# COMMAND ----------

audit_log_df = spark.createDataFrame(audit_log_data, audit_log_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC  STEP 1: FILTER RECENT REPORTS

# COMMAND ----------

try:
    recent_reports_df = radiology_reports_df.filter(
    (col("report_datetime") >= start_date) & (col("report_datetime") <= end_date)
    )
    if recent_reports_df.count() == 0:
        error_msg = "No recent reports found for the given period."
        raise ValueError(error_msg)
except Exception as e:

    error_msg = str(e)
    print(f"[{datetime.now()}] Job '{job_name}' (ID: {job_id}) FAILED. Error: {error_msg}")
    raise e
display(recent_reports_df)



# COMMAND ----------

# MAGIC %md
# MAGIC  STEP 2: JOIN WITH ORDERS

# COMMAND ----------

joined_data_df = recent_reports_df.alias("r").join(
        radiology_orders_df.alias("o"),
        col("r.order_id") == col("o.order_id"),
        "inner"
).select(
        col("r.report_id"),
        col("r.order_id"),
        col("r.report_datetime"),
        col("r.critical_finding_flag"),
        col("r.report_text"),
        col("o.patient_id"),
        col("o.modality"),
        col("o.order_datetime"),
        col("o.location_id"),
        col("o.attending_physician_id")
)
display(joined_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC STEP 3: DATA QUALITY CHECKS

# COMMAND ----------

dq_checked_df = joined_data_df.withColumn(
        "data_quality_issue",
        when(col("report_datetime").isNull(), lit("Missing Report Time"))
        .when(col("order_datetime").isNull(), lit("Missing Order Time"))
        .otherwise(lit(None))
    )
display(dq_checked_df)


# COMMAND ----------

clean_data_df = dq_checked_df.filter(col("data_quality_issue").isNull())
display(clean_data_df)  

# COMMAND ----------

# MAGIC %md
# MAGIC STEP 4: COMPUTE TURNAROUND TIME

# COMMAND ----------

with_turnaround_df = clean_data_df.withColumn(
        "turnaround_hrs",
        (datediff(col("report_datetime"), col("order_datetime")) * 24 * 60 +
         expr("minute(report_datetime) - minute(order_datetime)") +
         expr("hour(report_datetime) * 60 - hour(order_datetime) * 60")
        ) / 60.0
    )
display(with_turnaround_df) 

# COMMAND ----------

# MAGIC %md
# MAGIC STEP 5: MODALITY NORMALIZATION (EXAMPLE BUSINESS LOGIC)

# COMMAND ----------

normalized_df = with_turnaround_df.withColumn(
        "normalized_modality",
        when(col("modality").isin('XR', 'XRAY', 'X-RAY'), lit('XRAY'))
        .when(col("modality").isin('CT'), lit('CT'))
        .when(col("modality").isin('MRI', 'MR'), lit('MRI'))
        .when(col("modality").isin('US', 'ULTRASOUND'), lit('US'))
        .otherwise(lit('OTHER'))
    )
display(normalized_df)

# COMMAND ----------

# MAGIC %md
# MAGIC STEP 6: AGGREGATION

# COMMAND ----------

aggregated_summary_df = normalized_df.groupBy(
        "normalized_modality",
        "location_id",
        "attending_physician_id"
        ).agg(
        count("*").alias("total_studies"),
        avg("turnaround_hrs").alias("avg_turnaround_hrs"),
        sum(when(col("critical_finding_flag") == True, lit(1)).otherwise(lit(0))).alias("critical_findings")
    )
display(aggregated_summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC STEP 7: JOIN PHYSICIAN AND LOCATION INFO

# COMMAND ----------

final_summary_df = aggregated_summary_df.alias("s").join(
        physicians_df.alias("p"),
        col("s.attending_physician_id") == col("p.physician_id"),
        "left_outer"
    ).join(
        locations_df.alias("l"),
        col("s.location_id") == col("l.location_id"),
        "left_outer"
    ).select(
        col("s.normalized_modality").alias("modality"),
        col("l.location_name"),
        col("p.name").alias("physician_name"),
        col("s.total_studies"),
        col("s.avg_turnaround_hrs"),
        col("s.critical_findings")
    )
display(final_summary_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ARCHIVE PREVIOUS OUTPUT TABLE AND REPLACE THE NEW TABLE

# COMMAND ----------

print(f"[{today_timestamp}] Simulating archiving previous output to 'radiology_summary_archive'.")

# COMMAND ----------

print(f"[{today_timestamp}] Successfully generated the radiology summary report.")
final_summary_df.display()

# COMMAND ----------

    print(f"[{datetime.now()}] Job '{job_name}' (ID: {job_id}) SUCCESS.")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC  ERROR HANDLING AND FAILURE LOGGING

# COMMAND ----------

print(f"[{datetime.now()}] Temporary DataFrames cleanup handled automatically by Spark.")
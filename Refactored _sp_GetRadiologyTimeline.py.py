# Databricks notebook source
# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, max, count, lag, datediff, struct, udf
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, StructType
import xml.etree.ElementTree as ET
from datetime import datetime

class RadiologyDataProcessor:
    """
    A class to process radiology image data and generate patient timelines.
    """
    def __init__(self, spark):
        """
        Initializes the RadiologyDataProcessor with a SparkSession.

        Args:
            spark: The SparkSession.
        """
        self.spark = spark

    def read_radiology_data(self, data, schema):
        """
        Creates a Spark DataFrame from the given radiology data and schema.

        Args:
            data: A list of tuples representing the radiology data.
            schema: A StructType defining the schema of the radiology data.

        Returns:
            A Spark DataFrame.
        """
        df = self.spark.createDataFrame(data, schema)
        return df.withColumn("StudyDate", col("StudyDate").cast(DateType()))

    def read_patients_data(self, data, schema):
        """
        Creates a Spark DataFrame from the given patients data and schema.

        Args:
            data: A list of tuples representing the patients data.
            schema: A StructType defining the schema of the patients data.

        Returns:
            A Spark DataFrame.
        """
        df = self.spark.createDataFrame(data, schema)
        return df.withColumn("DOB", col("DOB").cast(DateType()))

    def filter_by_quality(self, radiology_df, min_quality):
        """
        Filters the radiology DataFrame based on the minimum quality score.

        Args:
            radiology_df: The input radiology DataFrame.
            min_quality: The minimum quality score.

        Returns:
            A filtered Spark DataFrame.
        """
        return radiology_df.filter(col("QualityScore") >= min_quality)

    def parse_report_xml(self, radiology_df):
        """
        Parses the XML reports in the radiology DataFrame to extract title and impression.

        Args:
            radiology_df: The radiology DataFrame containing the ReportXML column.

        Returns:
            A Spark DataFrame with 'ReportTitle' and 'Impression' columns.
        """
        def extract_report_info(xml_string):
            try:
                root = ET.fromstring(xml_string)
                title = root.findtext('Title')
                impression = root.findtext('Impression')
                return (title, impression)
            except Exception as e:
                return (None, None)

        extract_report_udf = udf(extract_report_info, StructType([
            StructField("ReportTitle", StringType()),
            StructField("Impression", StringType())
        ]))

        return radiology_df.withColumn("ReportDetails", extract_report_udf(col("ReportXML"))) \
                           .withColumn("ReportTitle", col("ReportDetails").getItem("ReportTitle")) \
                           .withColumn("Impression", col("ReportDetails").getItem("Impression")) \
                           .drop("ReportXML", "ReportDetails")

    def calculate_time_since_last_study(self, parsed_reports_df):
        """
        Calculates the time since the last study for each patient.

        Args:
            parsed_reports_df: DataFrame with parsed report information and StudyDate.

        Returns:
            DataFrame with 'TimeSinceLast' column.
        """
        window_spec = Window.partitionBy("PatientID").orderBy("StudyDate")
        return parsed_reports_df.withColumn(
            "TimeSinceLast",
            datediff(col("StudyDate"), lag(col("StudyDate"), 1).over(window_spec))
        )

    def pivot_modality_data(self, time_features_df):
        """
        Pivots the DataFrame to get the last study date for each modality and the total study count.

        Args:
            time_features_df: DataFrame with time-based features.

        Returns:
            DataFrame with 'Last_CT', 'Last_MRI', 'Last_XRAY', and 'Total_Studies' columns.
        """
        return time_features_df.groupBy("PatientID") \
                               .agg(
                                   max(when(col("Modality") == "CT", col("StudyDate"))).alias("Last_CT"),
                                   max(when(col("Modality") == "MRI", col("StudyDate"))).alias("Last_MRI"),
                                   max(when(col("Modality") == "XRAY", col("StudyDate"))).alias("Last_XRAY"),
                                   count("*").alias("Total_Studies")
                               )

    def join_with_patients(self, pivoted_df, patients_df):
        """
        Joins the pivoted radiology data with the patients DataFrame.

        Args:
            pivoted_df: DataFrame with pivoted radiology information.
            patients_df: DataFrame containing patient details.

        Returns:
            The final joined DataFrame with selected columns.
        """
        return pivoted_df.join(patients_df, "PatientID", "inner") \
                         .select("PatientID", "Name", "DOB", "Gender", "Last_CT", "Last_MRI", "Last_XRAY", "Total_Studies")



# COMMAND ----------


# Initialize SparkSession (assuming it's not already done in the Databricks notebook)
spark = SparkSession.builder.appName("RadiologyTimelineProcessor").getOrCreate()

# Widget to get the minimum quality score
dbutils.widgets.text("min_quality", "0", "Minimum Quality Score")
min_quality_widget = int(dbutils.widgets.get("min_quality"))

# Input Data
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

radiology_images_schema = StructType([
    StructField("PatientID", IntegerType(), False),
    StructField("Modality", StringType(), False),
    StructField("StudyDate", StringType(), False),
    StructField("StudyID", StringType(), False),
    StructField("QualityScore", IntegerType(), False),
    StructField("ReportXML", StringType(), False),
])

patients_data = [
    (1, 'John Doe', '1980-05-10', 'Male'),
    (2, 'Jane Smith', '1992-11-22', 'Female'),
    (3, 'Peter Jones', '1975-03-01', 'Male'),
]

patients_schema = StructType([
    StructField("PatientID", IntegerType(), False),
    StructField("Name", StringType(), False),
    StructField("DOB", StringType(), False),
    StructField("Gender", StringType(), False),
])

# Create an instance of the RadiologyDataProcessor
radiology_processor = RadiologyDataProcessor(spark)

# Load the DataFrames
radiology_df = radiology_processor.read_radiology_data(radiology_images_data, radiology_images_schema)
patients_df = radiology_processor.read_patients_data(patients_data, patients_schema)

# Process the data
filtered_df = radiology_processor.filter_by_quality(radiology_df, min_quality_widget)
parsed_reports_df = radiology_processor.parse_report_xml(filtered_df)
time_features_df = radiology_processor.calculate_time_since_last_study(parsed_reports_df)
pivoted_df = radiology_processor.pivot_modality_data(time_features_df)
final_df = radiology_processor.join_with_patients(pivoted_df, patients_df)

# Display the final DataFrame
final_df.display()

# Stop the SparkSession 
spark.stop()
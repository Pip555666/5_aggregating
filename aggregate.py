from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg, hour, year, lag, month
from pyspark.sql.window import Window
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import mysql.connector

# ✅ 환경 설정 함수
def setup():
    """
    - 한글 폰트 설정 (Matplotlib에서 한글이 깨지는 문제 방지)
    - 결과 저장 폴더 생성 (./chart)
    - Spark 세션 생성 및 설정 (로컬 모드에서 실행)
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 적용
    plt.rcParams['axes.unicode_minus'] = False  # 음수 부호 깨짐 방지
    os.makedirs('./chart', exist_ok=True)  # 결과 저장 폴더 생성

    spark = (SparkSession.builder
                .master("local")  # 로컬 환경에서 실행
                .appName("SentimentAggregation")  # Spark 애플리케이션 이름 설정
                .config("spark.ui.showConsoleProgress", "true")  # 실행 중 진행률 표시 설정
                .getOrCreate())
    
    spark.sparkContext.setLogLevel("INFO")  # 로그 레벨 설정 (INFO 레벨)
    return spark

# ✅ 데이터 로드 및 전처리 함수
def load_and_preprocess(spark, file_path):
    """
    - CSV 파일을 읽어와 Spark DataFrame으로 변환
    - 날짜, 시간, 연도 컬럼 추가
    - 공포탐욕지수 (prob_greed 값 * 100) 계산
    """
    df = spark.read.option("header", True).option("encoding", "UTF-8").csv(file_path, inferSchema=True)
    df = df.withColumn("date", to_date(col("time")))  # 날짜 컬럼 생성
    df = df.withColumn("hour", hour(col("time")))  # 시간 컬럼 생성
    df = df.withColumn("year", year(col("date")))  # 연도 컬럼 생성
    df = df.withColumn("공포탐욕지수", col("prob_greed") * 100)  # 공포탐욕지수 계산 (0~1 범위를 0~100 범위로 변환)
    return df

# ✅ 공포탐욕지수 평균 계산 및 저장 함수
def calculate_fear_greed(df, company):
    """
    - 시간대별 평균 공포탐욕지수 계산 및 저장 (연도별, 시간별 평균)
    - 월간 평균 공포탐욕지수 계산 및 저장 (월별 평균)
    """
    df_hourly = df.groupBy("year", "hour").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_hourly.toPandas().to_csv(f"./chart/{company}_hourly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")

    # 월 컬럼 추가 (YYYY-MM 형식)
    df = df.withColumn("month", col("date").substr(1, 7))
    df_monthly = df.groupBy("month").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_monthly_pandas = df_monthly.toPandas()
    df_monthly_pandas = df_monthly_pandas.dropna(subset=["month"])  # 결측값 제거
    df_monthly_pandas["month"] = df_monthly_pandas["month"].astype(str)  # 문자열 변환
    df_monthly_pandas = df_monthly_pandas.sort_values(by="month").reset_index(drop=True)  # 월 기준 정렬
    df_monthly_pandas.to_csv(f"./chart/{company}_monthly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")
    return df_monthly_pandas

# ✅ 공포탐욕지수 변화율 계산 함수
def calculate_change_rate(df, company):
    """
    - 시간대별 공포탐욕지수 변화율 계산
    - Spark Window 함수를 이용해 이전 값과 차이 계산
    - 결과 CSV 저장
    """
    window_spec = Window.partitionBy("year").orderBy("hour")  # 연도별 정렬 기준 설정
    df = df.withColumn("feargreed_diff", col("공포탐욕지수") - lag(col("공포탐욕지수"), 1).over(window_spec))  # 변화량 계산
    df_change_rate = df.groupBy("year", "hour").agg(avg("feargreed_diff").alias("변화율"))  # 시간대별 평균 변화율 계산
    df_change_rate_pandas = df_change_rate.toPandas()
    df_change_rate_pandas = df_change_rate_pandas.sort_values(by=["year", "hour"]).reset_index(drop=True)  # 정렬 및 인덱스 초기화
    df_change_rate_pandas.to_csv(f"./chart/{company}_feargreed_change_rate.csv", index=False, encoding="utf-8-sig")
    return df_change_rate_pandas

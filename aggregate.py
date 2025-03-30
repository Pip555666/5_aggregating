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
    - 한글 폰트 설정 (Matplotlib에서 한글 깨짐 방지)
    - 결과 저장 폴더 ('./chart') 생성 (없으면 생성)
    - Spark 세션 생성 및 설정
    """
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    os.makedirs('./chart', exist_ok=True)  # 결과 저장 폴더 생성

    # Spark 세션 생성 및 기본 설정
    spark = (SparkSession.builder
                .master("local")  # 로컬 실행
                .appName("SentimentAggregation")  # Spark 애플리케이션 이름
                .config("spark.ui.showConsoleProgress", "true")  # 콘솔에서 진행 상황 출력 설정
                .getOrCreate())

    spark.sparkContext.setLogLevel("INFO")  # 로그 레벨 설정
    return spark

# ✅ 데이터 로드 및 전처리 함수
def load_and_preprocess(spark, file_path):
    """
    - CSV 파일을 Spark DataFrame으로 로드
    - 날짜(date), 시간(hour), 연도(year) 컬럼 추가
    - 공포탐욕지수(Fear & Greed Index) 계산 (prob_greed * 100)
    """
    df = spark.read.option("header", True).option("encoding", "UTF-8").csv(file_path, inferSchema=True)
    df = df.withColumn("date", to_date(col("time")))  # 날짜 변환
    df = df.withColumn("hour", hour(col("time")))  # 시간 추출
    df = df.withColumn("year", year(col("date")))  # 연도 추출
    df = df.withColumn("공포탐욕지수", col("prob_greed") * 100)  # 공포탐욕지수 계산
    return df

# ✅ 공포탐욕지수 평균 계산 및 저장 함수
def calculate_fear_greed(df, company):
    """
    - 시간대별 평균 공포탐욕지수 계산 및 CSV 저장
    - 월간 평균 공포탐욕지수 계산 및 CSV 저장
    """
    # 시간대별 평균 계산
    df_hourly = df.groupBy("year", "hour").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_hourly.toPandas().to_csv(f"./chart/{company}_hourly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")

    # 월별 데이터 변환 (MM 형태)
    df = df.withColumn("month", col("date").substr(1, 7))
    
    # 월별 평균 계산
    df_monthly = df.groupBy("month").agg(avg("공포탐욕지수").alias("평균_공포탐욕지수"))
    df_monthly_pandas = df_monthly.toPandas()

    # 결측값 제거 및 정렬
    df_monthly_pandas = df_monthly_pandas.dropna(subset=["month"])  
    df_monthly_pandas["month"] = df_monthly_pandas["month"].astype(str)
    df_monthly_pandas = df_monthly_pandas.sort_values(by="month").reset_index(drop=True)

    df_monthly_pandas.to_csv(f"./chart/{company}_monthly_feargreed_score_bert.csv", index=False, encoding="utf-8-sig")
    return df_monthly_pandas

# ✅ 공포탐욕지수 변화율 계산 함수
def calculate_change_rate(df, company):
    """
    - 시간대별 공포탐욕지수 변화율 계산
    - 결과 CSV 저장
    """
    window_spec = Window.partitionBy("year").orderBy("hour")
    df = df.withColumn("feargreed_diff", col("공포탐욕지수") - lag(col("공포탐욕지수"), 1).over(window_spec))

    df_change_rate = df.groupBy("year", "hour").agg(avg("feargreed_diff").alias("변화율"))
    df_change_rate_pandas = df_change_rate.toPandas()

    # 정렬 후 저장
    df_change_rate_pandas = df_change_rate_pandas.sort_values(by=["year", "hour"]).reset_index(drop=True)
    df_change_rate_pandas.to_csv(f"./chart/{company}_feargreed_change_rate.csv", index=False, encoding="utf-8-sig")
    return df_change_rate_pandas

# ✅ 시각화 함수
def save_plots(df_change_rate_pandas, df_monthly_pandas, company):
    """
    - 공포탐욕지수 변화율 그래프 저장
    - 월간 평균 공포탐욕지수 그래프 저장
    """
    years = df_change_rate_pandas["year"].unique()

    for year in years:
        df_yearly = df_change_rate_pandas[df_change_rate_pandas["year"] == year]

        plt.figure(figsize=(12, 6))
        plt.plot(df_yearly["hour"], df_yearly["변화율"], marker='o', linestyle='-', color='red', label=f'{year}년 공포탐욕지수 변화율')
        plt.axhline(0, color='gray', linestyle='--', label='기준선')
        plt.title(f"{company} {year}년 시간대별 공포탐욕지수 변화율")
        plt.xlabel("시간")
        plt.ylabel("변화율")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./chart/{company}_{year}_fear_and_greed_change_rate.png")
        plt.close()

    # 월간 평균 그래프 저장
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly_pandas["month"], df_monthly_pandas["평균_공포탐욕지수"], marker='o', linestyle='-', color='blue', label='월간 평균 공포탐욕지수')
    plt.title(f"{company} 월간 평균 공포탐욕지수")
    plt.xlabel("월")
    plt.ylabel("평균 공포탐욕지수")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./chart/{company}_monthly_fear_and_greed.png")
    plt.close()

# ✅ 실행 코드
spark = setup()

# MySQL 연결 설정
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Password',
    'database': 'stock_analysis'
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

try:
    print("MySQL 데이터베이스 연결 성공")
    companies = ["samsung", "apple", "nvidia", "skhynix"]
    
    for company in companies:
        print(f"--- 현재 처리 중인 회사: {company} ---")
        try:
            file_path = f"file:///D:/Project/{company}_predict_bert.csv"
            df = load_and_preprocess(spark, file_path)
            df_monthly_pandas = calculate_fear_greed(df, company)
            df_change_rate_pandas = calculate_change_rate(df, company)
            save_plots(df_change_rate_pandas, df_monthly_pandas, company)

            print(f"{company} 데이터 처리 완료")
        except Exception as e:
            print(f"{company} 처리 중 오류 발생: {e}")
            continue  # 오류 발생 시 다음 회사로 넘어감
except mysql.connector.Error as err:
    print(f"MySQL 오류 발생: {err}")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL 연결 종료")

spark.stop()

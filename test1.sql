CREATE TABLE sentiment_indicators (
    stock_code VARCHAR(10) NOT NULL, -- 종목 코드
    date DATE NOT NULL,             -- 날짜
    fear_ratio DECIMAL(5, 4) NOT NULL, -- 공포 비율 (예: 0.0000 ~ 1.0000)
    neutral_ratio DECIMAL(5, 4) NOT NULL, -- 중립 비율
    greed_ratio DECIMAL(5, 4) NOT NULL, -- 탐욕 비율
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_code, date)
);
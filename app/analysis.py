from flask import Flask, request, jsonify
import requests
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

SERVICE_KEY = 'mUglWToAvQmeYwqfOR1UwESkdwoUdkYmZTS5lWDf5pEGuh1gvgyfIy%2BtFKzTNsFYqE%2BM0a6NlwJGtxyhBV63sQ%3D%3D'

def get_stock_code(stock_name):
    """한국 주식 종목명을 입력받아 종목 코드 조회"""
    url = f"https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo?serviceKey={SERVICE_KEY}&numOfRows=1&pageNo=1&resultType=json&itmsNm={stock_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        items = data.get('response', {}).get('body', {}).get('items', {}).get('item', [])
        if items:
            stock_code = items[0].get('srtnCd', '')  # 짧은 종목 코드
            market_type = items[0].get('mrktCtg', '')  # 시장 구분
            suffix = '.KS' if market_type == 'KOSPI' else '.KQ'
            return f"{stock_code}{suffix}"
    return None

def get_stock_data(stock_name, country, period='5y', interval='monthly'):
    """yfinance로 주식 데이터 가져오기"""
    ticker = get_stock_code(stock_name) if country == 'KR' else stock_name
    if not ticker:
        return None
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if interval == 'monthly':
        df_resampled = df['Close'].resample('M').last().reset_index()
    elif interval == 'semi_monthly':
        df = df.reset_index()
        df['Day'] = df['Date'].dt.day
        mid_month = df[df['Day'] == 15]
        month_end = df[df['Date'] == df.groupby(df['Date'].dt.to_period('M'))['Date'].transform('max')]
        df_resampled = pd.concat([mid_month, month_end]).sort_values('Date').reset_index(drop=True)
    else:
        return None
    return df_resampled[['Date', 'Close']]

def cal_increase(stock_name, country, interval='monthly'):
    """주식 데이터 증감률 계산"""
    df = get_stock_data(stock_name, country, interval=interval)
    if df is not None:
        df['Increase'] = df['Close'].diff()
        return df.dropna().reset_index(drop=True)
    return None

def plot_stats_dual_axis(stats_df, interval):
    """
    통계 데이터를 이중 축으로 시각화.
    - 왼쪽 y축: Median (Increase)
    - 오른쪽 y축: Increase Count
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))
    x_labels = stats_df['Month'] if interval == 'monthly' else stats_df['Month-Period']
    ax1.set_xlabel('Month' if interval == 'monthly' else 'Month-Period', fontsize=12)
    ax1.set_ylabel('Median (Increase)', fontsize=12, color='red')
    ax1.plot(x_labels, stats_df['Median'], marker='o', color='red', label='Median (Increase)', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Increase Count', fontsize=12, color='darkorange')
    ax2.bar(x_labels, stats_df['Increase Count'], alpha=0.6, label='Increase Count', width=0.4, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='black')
    plt.title('Monthly Statistics' if interval == 'monthly' else 'Semi-Monthly Statistics', fontsize=16)
    plt.xticks(rotation=45, fontsize=10)
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return img_base64

def cal_data(stock_name, country='US', interval='monthly'):
    """
    주어진 종목 데이터를 기반으로 월별 또는 월중(15일) 및 월말 통계량 계산 및 그래프 출력.
    """
    if interval == 'monthly':
        df_monthly = cal_increase(stock_name, country=country, interval='monthly')
        if df_monthly is None:
            return None, None
        df_monthly['Month'] = df_monthly['Date'].dt.month
        monthly_stats = df_monthly.groupby('Month')['Increase'].agg(
            Max='max',
            Min='min',
            Median='median',
            Mean='mean',
            Std='std',
            Variance='var',
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
        )
        positive_counts = df_monthly[df_monthly['Increase'] > 0].groupby('Month')['Increase'].count()
        monthly_stats = monthly_stats.join(positive_counts.rename('Increase Count')).fillna(0).reset_index()
        monthly_stats['Increase Count'] = monthly_stats['Increase Count'].astype(int)
        graph_image = plot_stats_dual_axis(monthly_stats, interval='monthly')
        return monthly_stats, graph_image
    return None, None

def predict_increase(stock_name, country='US', interval='monthly'):
    """SARIMA 모델을 사용해 증감률 예측"""
    if interval == 'monthly':
        df_monthly = cal_increase(stock_name, country=country, interval='monthly')
        if df_monthly is None:
            return None
        model = SARIMAX(df_monthly['Increase'], order=(2, 1, 1), seasonal_order=(0, 1, 1, 12))
        monthly_model_fit = model.fit(disp=False)
        monthly_forecast = monthly_model_fit.forecast(steps=12)
        monthly_start_date = df_monthly['Date'].iloc[-1]
        monthly_forecast_dates = pd.date_range(start=monthly_start_date + pd.DateOffset(months=1), periods=12, freq='M')
        return pd.DataFrame({'Date': monthly_forecast_dates, 'Predicted_Increase': monthly_forecast})
    elif interval == 'semi_monthly':
        df_semi_monthly = cal_increase(stock_name, country=country, interval='semi_monthly')
        if df_semi_monthly is None:
            return None
        df_semi_monthly['Date'] = pd.to_datetime(df_semi_monthly['Date'])
        df_semi_monthly.set_index('Date', inplace=True)
        model = SARIMAX(df_semi_monthly['Increase'], order=(2, 1, 1), seasonal_order=(0, 1, 1, 12))
        semi_monthly_model_fit = model.fit(disp=False)
        semi_monthly_forecast = semi_monthly_model_fit.forecast(steps=24)
        last_date = df_semi_monthly.index[-1]
        forecast_dates = []
        for i in range(1, 13):
            mid_month = (last_date + pd.DateOffset(months=i)).replace(day=15)
            end_month = (last_date + pd.DateOffset(months=i + 1)).replace(day=1) - pd.DateOffset(days=1)
            forecast_dates.extend([mid_month, end_month])
        return pd.DataFrame({'Date': forecast_dates, 'Predicted_Increase': semi_monthly_forecast})
    return None

@app.route('/calculate_stats', methods=['GET'])
def api_calculate_stats():
    """
    주어진 종목 데이터의 통계량 계산 및 시각화 API.
    """
    stock_name = request.args.get('stock_name')
    country = request.args.get('country', 'US')
    interval = request.args.get('interval', 'monthly')
    stats, graph_image = cal_data(stock_name, country, interval)
    if stats is not None:
        return jsonify({
            'stats': stats.to_dict(orient='records'),
            'graph_image': graph_image
        })
    return jsonify({'error': 'Failed to calculate stats'}), 500

@app.route('/predict_increase', methods=['GET'])
def api_predict_increase():
    """
    주어진 종목 데이터의 예측 결과 반환 API
    """
    stock_name = request.args.get('stock_name')
    country = request.args.get('country', 'US')
    interval = request.args.get('interval', 'monthly')
    forecast_df = predict_increase(stock_name, country, interval)
    if forecast_df is not None:
        return forecast_df.to_json(orient='records', date_format='iso')
    return jsonify({'error': 'Failed to predict data'}), 500

if __name__ == '__main__':
    app.run(debug=True)

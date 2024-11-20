from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
from app.analysis import cal_data, predict_increase, get_stock_code, get_stock_data

main = Blueprint('main', __name__, url_prefix='/api')

def get_stock_info(stock_name):
    stock_code = get_stock_code(stock_name)
    if not stock_code:
        return None, {"error": "Invalid stock name"}
    return stock_code, None

@main.route('/stock/search', methods=['GET'])
@cross_origin()
def search_stock():
    try:
        keyword = request.args.get('keyword')
        if not keyword:
            return jsonify({"error": "Keyword is required"}), 400
        result = get_stock_code(keyword)
        if result is None:
            return jsonify({"error": "Stock not found"}), 404
        return jsonify({"code": result, "name": keyword})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/stock/analyze', methods=['GET'])
@cross_origin()
def analyze_stock():
    try:
        stock_name = request.args.get('stockName')
        if not stock_name:
            return jsonify({"error": "Stock name is required"}), 400
        
        stock_code, error = get_stock_info(stock_name)
        if error:
            return jsonify(error), 404
        
        stock_data = get_stock_data(stock_code, country='KR', period='5y', interval='monthly')
        if stock_data is None:
            return jsonify({"error": "Failed to fetch stock data"}), 500
        
        analysis_result = cal_data(stock_code)
        if "error" in analysis_result:
            return jsonify({"error": analysis_result["error"]}), 500
        
        return jsonify({
            "stockName": stock_name,
            "stockCode": stock_code,
            "statistics": analysis_result["statistics"],
            "visualization": analysis_result["visualization"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/stock/predict', methods=['GET'])
@cross_origin()
def predict_stock():
    try:
        stock_name = request.args.get('stockName')
        if not stock_name:
            return jsonify({"error": "Stock name is required"}), 400
        
        stock_code, error = get_stock_info(stock_name)
        if error:
            return jsonify(error), 404
        
        prediction_result = predict_increase(stock_code)
        if "error" in prediction_result:
            return jsonify({"error": prediction_result["error"]}), 500
            
        return jsonify({
            "stockName": stock_name,
            "stockCode": stock_code,
            "prediction": prediction_result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main.route('/news', methods=['GET'])
@cross_origin()
def get_news():
    stock_name = request.args.get('stockName')
    if not stock_name:
        return jsonify({"error": "Stock name is required"}), 400
    
    # 여기에 뉴스 데이터를 가져오는 로직을 추가하세요
    # 예시로 더미 데이터를 반환합니다.
    news_data = [
        {"title": "뉴스 제목 1", "date": "2023-01-01"},
        {"title": "뉴스 제목 2", "date": "2023-01-02"},
    ]
    
    return jsonify(news_data)
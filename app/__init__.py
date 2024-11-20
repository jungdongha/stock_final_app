from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    # CORS 설정을 조금 더 명확하게
    CORS(app, resources={r"/api/*": {"origins": "*"}})  # 필요한 경우 origins에 특정 도메인을 추가

    from app.routes import main
    app.register_blueprint(main)

    return app

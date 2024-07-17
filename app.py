import json
import time
import os

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import chat_service

app = Flask(__name__)
cors = CORS(app)
chat_res = chat_service.chatbot_service()
load_dotenv()
os.environ['OPENAI_API_KEY']
@app.route('/', methods=["GET"])
def chat():

    return render_template('chat.html')


@app.route('/chat', methods=["POST"])
def chat_rag():

    user_message = request.json
    start = time.time()
    response_message = generate_bot_response(user_message, 'chat')

    print(f"답변시간 : {time.time() - start:.5f} sec")

    # 한국어 인코딩
    res = json.dumps(response_message, ensure_ascii=False).encode('utf8')
    return Response(res, content_type='application/json; charset=utf-8')


@app.route('/search', methods=["POST"])
def similarity_search():
    user_message = request.json

    # 사용자 질문과 유사도가 높은 검색된 상위 3개의 문장, 유사도 받아오기
    source = generate_bot_response(user_message, 'search')
    print(source)

    return source.to_json(orient='split')


def generate_bot_response(user_message, type):
    # 간단한 봇 응답 로직 (필요에 따라 수정 가능)

    answer = ''
    if user_message == "안녕":
        answer = "안녕하세요! 무엇을 도와드릴까요?"

    elif user_message == "!도움말":
        answer = "명령어 목록: !구매, !다시, !공유"

    elif user_message == "!임베딩":
        chat_res.caching_flies()
        answer = "자료 학습 완료!"

    else:
        if type == "chat":
            answer = chat_res.caching_embeds(user_message)

        if type == "search":
            # string이 아닌 dataframe을 반환
            source = chat_res.caching_similar_search(user_message)
            return source

    return answer


if __name__ == '__main__':
    app.run()

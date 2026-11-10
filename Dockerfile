# 검색(임베더·리랭커, CPU)으로 근거를 찾고 생성은 vLLM 컨테이너에 HTTP로 맡기는 FastAPI 앱.
FROM python:3.10-slim

WORKDIR /app

# slim 기본 pip가 오래돼 CPU 휠을 못 집고 소스 빌드로 떨어지므로 먼저 pip를 올린다.
RUN pip install --no-cache-dir --upgrade pip
# 임베더·리랭커용 torch는 CPU 전용 휠로 설치한다(GPU는 vLLM 컨테이너만 쓴다).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 앱 코드와 논문 데이터만 넣는다(코퍼스 인덱스 등 대용량은 볼륨으로 마운트).
COPY src ./src
COPY data/papers ./data/papers

EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]

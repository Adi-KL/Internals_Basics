FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models/ models/
COPY src/predict_cli.py src/predict_cli.py
ENTRYPOINT ["python", "src/predict_cli.py"]
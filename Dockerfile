FROM python:3.12-slim
WORKDIR /app
# Корневые сертификаты Минцифры — нужны для TLS к botapi.max.ru и API Yandex Cloud.
COPY certs/*.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "src.bot.max_bot"]

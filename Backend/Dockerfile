FROM python:3.10

WORKDIR /app

COPY requirements-runtime.txt .

RUN pip install --upgrade pip && pip install -r requirements-runtime.txt

COPY app ./app

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.api.main:app", "--bind", "0.0.0.0:8000"]

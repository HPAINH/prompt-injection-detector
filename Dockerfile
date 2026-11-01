FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIG_ROOT=/app \
    PIG_DETECTOR=semantic \
    HF_HOME=/app/.hf

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e .

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "pig.service.api:app", "--host", "0.0.0.0", "--port", "8000"]

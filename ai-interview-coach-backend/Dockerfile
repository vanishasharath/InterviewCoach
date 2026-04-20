FROM python:3.9
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV DOCKER_BUILDKIT=0
LABEL hf.space.cache="false"


RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH=/app

WORKDIR /app

COPY backend/requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


RUN python -m spacy download en_core_web_sm
# (or en_core_web_lg if you chose lg)

COPY backend /app/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

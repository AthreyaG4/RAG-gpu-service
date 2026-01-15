FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV USER=appuser
RUN useradd -m -s /bin/bash $USER

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R $USER:$USER /app

USER $USER

EXPOSE 8000

CMD ["uvicorn", "script:app", "--host", "0.0.0.0", "--port", "8000"]
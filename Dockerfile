FROM python:3.11.9-bullseye

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY streamlit ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "lostark_dmg_analysis.py"]
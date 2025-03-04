FROM python:3.13.1
WORKDIR /app
# Install system libraries for audio (for Speech SDK)
RUN apt-get update && apt-get install -y \
    libasound2 \
    libasound2-plugins \
    alsa-utils \
    alsa-oss \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./app.py
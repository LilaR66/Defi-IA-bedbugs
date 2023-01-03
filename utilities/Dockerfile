# We use a minimal Ubuntu image optimized for Docker
FROM ubuntu:latest

# Set up for your local zone an UTC information
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Our own build instructions
RUN apt-get update && apt-get install -y \
#git \
#wget \
#nano \
python3-pip

# Additional librairies:
RUN pip install gradio==3.4
RUN pip install xgboost==1.0.1
RUN pip install numpy==1.23.3
RUN pip install pandas==1.5.0
RUN pip install scikit-learn==1.1.2
RUN pip install seaborn==0.12.1
RUN pip install sklearn==0.0
RUN pip install category-encoders==2.5.1.post0
RUN pip install matplotlib==3.6.0
RUN pip install scipy==1.9.1
RUN pip install pickle5
RUN pip install xgboost==1.0.1
RUN pip install catboost
RUN pip install lightgbm

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

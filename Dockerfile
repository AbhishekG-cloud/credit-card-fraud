FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y

RUN pip install -r requirement.txt
CMD ["python","app.py"]
FROM python:3.9
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt
EXPOSE 8051
CMD ["streamlit", "run", "main.py", "--server.port=8051", "--server.address=0.0.0.0"]

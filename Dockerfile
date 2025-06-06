FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN chmod +x startup.sh

EXPOSE 8000

CMD ["./startup.sh"]

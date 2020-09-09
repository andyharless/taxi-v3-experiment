FROM python:3.8.5

RUN pip install numpy==1.19
RUN pip install gym==0.17

COPY ./*.py /usr/src/
WORKDIR /usr/src

CMD ["python", "main.py"]


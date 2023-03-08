FROM python:3.9.7
ADD client.py /tmp
ADD data.csv /tmp
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
CMD ["python3","/tmp/client.py"]

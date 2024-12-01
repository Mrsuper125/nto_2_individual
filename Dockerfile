FROM odsai/python-gpu:latest


RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install ultralytics
RUN pip install dill

RUN mkdir /tmp/Ultralytics

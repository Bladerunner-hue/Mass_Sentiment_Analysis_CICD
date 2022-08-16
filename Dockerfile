# For more information, please refer to https://aka.ms/vscode-docker-python
FROM centos:latest
RUN yum install net-tools -y
RUN yum install httpd -y 
RUN install python3 -y 
COPY requirements.txt /home
RUN pip3 install -t /home/requirements.txt
COPY Web_App Sentiment_Analysis
WORKDIR "/src/Web_App/Sentiment_Analysis"
ENTRYPOINT [ "python3","flask_sentiment_analysis_app.py"]
EXPOSE 3000 5050
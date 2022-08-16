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
RUN curl -fsSLO https://get.docker.com/builds/Linux/x86_64/docker-17.04.0-ce.tgz \
  && tar xzvf docker-17.04.0-ce.tgz \
  && mv docker/docker /usr/local/bin \
  && rm -r docker docker-17.04.0-ce.tgzs
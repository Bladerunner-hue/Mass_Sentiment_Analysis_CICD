# For more information, please refer to https://aka.ms/vscode-docker-python
FROM centos:latest
RUN yum install net-tools -y
RUN yum install httpd -y 
RUN install python3 -y 
COPY requirements.txt /home
RUN pip3 install -t /home/requirements.txt
COPY 
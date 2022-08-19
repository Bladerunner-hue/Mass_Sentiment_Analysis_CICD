# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./localConsole.py" ]


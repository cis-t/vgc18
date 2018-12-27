FROM joemzhao/base-py

MAINTAINER joemzhao <mengjie.zhao@outlook.com>

USER root

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt install -y python-pip
RUN pip install -U pip

RUN pip install nltk \
				numpy \ 
				overrides \
				scikit-learn \
				scipy \
				tqdm
				
RUN pip install --ignore-installed six				
RUN pip install tornado==4.* # skip tornado compatibly issue
RUN python -m pip install jupyter
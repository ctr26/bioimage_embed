#!make
include .env
include secrets.env
export

download.data:
	kaggle competitions download -c data-science-bowl-2018

test:
	pytest

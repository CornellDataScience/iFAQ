#!/usr/bin/env bash
cd ../
cp env.template .env

sudo apt install virtualenv
virtualenv -p python3 venv

pip install -r requirements.txt
echo "source `which activate.sh`" >> ~/.bashrc

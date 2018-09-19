cd iFAQ

sudo apt install virtualenv
virtualenv -p python3 venv

pip install autoenv
echo "source `which activate.sh`" >> ~/.bashrc

pip install python-dotenv
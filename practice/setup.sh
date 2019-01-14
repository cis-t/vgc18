# setting up environment
pip install virtualenv
virtualenv --no-site-packages ./sentiment-analaysis-vgc
wait

# install requirements
source ./sentiment-analaysis-vgc/bin/activate
python -m pip install jupyter
pip install -r requirements.txt

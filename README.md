# fNIRS perceptual analysis

- Copy files (link OpenNeuro dataset when ready) into `data`
- Copy fOLD specificity files into `data`

~~~
cd data
git clone https://github.com/nirx/fOLD-public.git
cd ..
~~~

Analysis was run with Python 3.14.2

~~~
python -m venv venv
source venv/bin/activate
pip install -r requirements_max.txt
~~~

Currently using a forked version of mne-nirs, supporting saving auxilliary data (such as accelerometer data) to BIDS datasets, as well as implementing TCCA.

~~~
git clone https://github.com/alkvi/mne-nirs.git
cd mne-nirs
git checkout dev
pip install -e .
~~~

# fNIRS perceptual analysis

- Copy files (link OpenNeuro dataset when ready) into `data`
- Copy fOLD specificity files into `data`

~~~
cd data
git clone https://github.com/nirx/fOLD-public.git
cd ..
~~~

## Preparing Python

The notebook is run in Python 3.14.

Environment and Python is handled by `uv`

MacOS, via [Homebrew](https://brew.sh):

~~~
brew install uv
~~~

Windows, via [WinGet](https://github.com/microsoft/winget-cli):

~~~
winget install --id=Astral-sh.uv

or just via powershell

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
~~~

Linux:

~~~
curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

Install Python 3.14.2:

~~~
uv python install 3.14.2
~~~

## Installing dependencies

MacOS / Linux:

~~~
uv venv --python 3.14.2
source .venv/bin/activate
uv pip install -r requirements.txt
~~~

Windows:

~~~
uv venv --python 3.14.2
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
~~~

## Preparing MNE-NIRS

Currently using a forked version of mne-nirs, supporting saving auxilliary data (such as accelerometer data) to BIDS datasets, as well as implementing TCCA.

~~~
git clone https://github.com/alkvi/mne-nirs.git
cd mne-nirs
git checkout dev
uv pip install -e .
~~~

## Running the notebook

Go back to the folder where this repository is stored.

~~~
jupyter notebook
~~~

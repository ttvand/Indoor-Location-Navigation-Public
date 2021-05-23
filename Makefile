setup:
	python3 -m venv venv
	./venv/bin/pip3 install -U pip
	./venv/bin/pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
	cat requirements.txt | xargs -n 1 ./venv/bin/pip3 install

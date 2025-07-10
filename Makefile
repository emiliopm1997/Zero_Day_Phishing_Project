beautify:
	python -m isort zdpd_model
	python -m flake8

rebuild_package:
	rm -rf dist
	rm -rf zdpd_model.egg-info
	python ./setup.py sdist --formats=gztar

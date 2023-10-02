import os


def inicial_config(path):

	if not os.path.isdir(path):
		os.mkdir(path)


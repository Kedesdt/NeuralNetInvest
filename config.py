import os


def inicial_config(path, nome):

	if not os.path.isdir(path):
		os.mkdir(path)

	if not os.path.isdir(path + "/" + nome):
		os.mkdir(path + "/" + nome)


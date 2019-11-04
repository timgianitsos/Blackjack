# Blackjack

## Authors

Daniel Mendoza

Maika Isogawa

[Tim Gianitsos](https://github.com/timgianitsos)

## Setup

1. Ensure `Python` version 3.8 is installed.
1. Ensure `pipenv` is installed
1. While in the project directory, run the following command. This will generate a virtual environment called `.venv` in the current directory that will contain the `Python` dependencies for this project.<sup id="a1">[1](#f1)</sup>
	```bash
	PIPENV_VENV_IN_PROJECT=true pipenv --python 3.8
	```
1. This will activate the virtual environment. After activation, running `Python` commands will ignore the system-level `Python` version & packages, and only use the packages from the virtual environment.
	```bash
	pipenv shell
	```
1. This will install all<sup id="a2">[2](#f2)</sup> the requisite packages into the virtual environment. The packages and exact versions are specified in `Pipfile.lock`.
	```bash
	pipenv install --dev
	```
1. This will exit the virtual environment i.e. it restores the system-level `Python` configurations to your shell.
	```bash
	exit
	```
1. Whenever you want to resume working on the project, run the following while in the project directory to activate the virtual environment again. This will make `Python` use the version and dependencies in the project's virtual environment.
	```bash
	pipenv shell
	```

## Notes

<b id="f1">1)</b> The `pipenv` tool works by making a project-specific directory called a virtual environment that hold the dependencies for that project. After a virtual environment is activated, newly installed dependencies will automatically go into the virtual environment instead of being placed among your system-level `Python` packages.  
Setting the `PIPENV_VENV_IN_PROJECT` variable to true will indicate to `pipenv` to make this virtual environment within the same directory as the project so that all the files corresponding to a project can be in the same place. This is [not default behavior](https://github.com/pypa/pipenv/issues/1382) (e.g. on Mac, the environments will normally be placed in `~/.local/share/virtualenvs/` by default). [↩](#a1)

<b id="f2">2)</b> Using `--dev` ensures that even development dependencies will be installed (dev dependencies may include testing and linting frameworks which are not necessary for normal execution of the code). After installation, you can find all dependencies (both dev dependencies and regular dependencies) in `<path to virtual environment>/lib/python3.8/site-packages/`. [↩](#a2)


DATA_DIR=data
SCRIPTS_DIR=scripts
MODELS_DIR=models
RESULTS_DIR=results
CONDA_HOME=C:\Users\lyona\anaconda3
CONDA_ENV=mlops_venv
VENV_DIR=venv
DOCKER_ENV=$(shell if [ -f /.dockerenv ]; then echo "true"; else echo "false"; fi)

.PHONY: all env_conda env_pip data train evaluate clean

env_conda:
	@if [ "$(DOCKER_ENV)" = "true" ]; then \
		$(MAKE) env_pip; \
	else \
		if [ ! -d "$(CONDA_HOME)" ]; then \
			echo "Conda tidak ditemukan. Menggunakan virtualenv..." && $(MAKE) env_pip; \
		else \
			conda info --envs | grep $(CONDA_ENV) || ( \
				echo "Membuat environment conda..." && conda create --name $(CONDA_ENV) python=3.8 -y && \
				. $(CONDA_HOME)/bin/activate $(CONDA_ENV) && pip install -r requirements.txt \
			); \
		fi \
	fi

env_pip:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python -m venv $(VENV_DIR) && . $(VENV_DIR)/bin/activate && pip install -r requirements.txt; \
	fi

all: env_conda
	@if [ "$(DOCKER_ENV)" = "true" ]; then \
		. venv/bin/activate && \
		python $(SCRIPTS_DIR)/data_prep.py && \
		python $(SCRIPTS_DIR)/train_model.py && \
		python $(SCRIPTS_DIR)/evaluate_model.py; \
		python $(SCRIPTS_DIR)/deploy_model.py; \
	else \
		if [ ! -d "$(CONDA_HOME)" ]; then \
			. $(VENV_DIR)/bin/activate && \
			python $(SCRIPTS_DIR)/data_prep.py && \
			python $(SCRIPTS_DIR)/train_model.py && \
			python $(SCRIPTS_DIR)/evaluate_model.py; \
			python $(SCRIPTS_DIR)/deploy_model.py; \
		else \
			. $(CONDA_HOME)/bin/activate $(CONDA_ENV) && \
			python $(SCRIPTS_DIR)/data_prep.py && \
			python $(SCRIPTS_DIR)/train_model.py && \
			python $(SCRIPTS_DIR)/evaluate_model.py; \
			python $(SCRIPTS_DIR)/deploy_model.py; \
		fi \
	fi

data: env_conda
	@if not exist "$(CONDA_HOME)" ( \
		$(VENV_DIR)\Scripts\activate && python $(SCRIPTS_DIR)/data_prep.py; \
	) else ( \
		conda activate $(CONDA_ENV) && python $(SCRIPTS_DIR)/data_prep.py \
	)

train: env_conda
	@if not exist "$(CONDA_HOME)" ( \
		$(VENV_DIR)\Scripts\activate && python $(SCRIPTS_DIR)/train_model.py; \
	) else ( \
		conda activate $(CONDA_ENV) && python $(SCRIPTS_DIR)/train_model.py \
	)

evaluate: env_conda
	@if not exist "$(CONDA_HOME)" ( \
		$(VENV_DIR)\Scripts\activate && python $(SCRIPTS_DIR)/evaluate_model.py; \
	) else ( \
		conda activate $(CONDA_ENV) && python $(SCRIPTS_DIR)/evaluate_model.py \
	)

deploy: env_conda
	@if not exist "$(CONDA_HOME)" ( \
		$(VENV_DIR)\Scripts\activate && python $(SCRIPTS_DIR)/deploy_model.py; \
	) else ( \
		conda activate $(CONDA_ENV) && python $(SCRIPTS_DIR)/deploy_model.py \
	)

clean:
	del /Q $(DATA_DIR)\*.csv $(MODELS_DIR)\*.pkl $(RESULTS_DIR)\*.txt
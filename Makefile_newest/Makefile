DATA_DIR=data
SCRIPTS_DIR=scripts
MODELS_DIR=models
RESULTS_DIR=results
CONDA_HOME=C:\Users\lyona\anaconda3
CONDA_ENV=mlops_venv
VENV_DIR=venv

.PHONY: all env_conda env_pip data train evaluate clean

env_conda:
	@if not exist "$(CONDA_HOME)" ( \
		echo "Conda tidak ditemukan. Menggunakan virtualenv..." && $(MAKE) env_pip; \
	) else ( \
		conda info --envs | findstr $(CONDA_ENV) >nul || ( \
			echo "Membuat environment conda..." && conda create --name $(CONDA_ENV) python=3.8 -y && \
			conda activate $(CONDA_ENV) && pip install -r requirements.txt \
		) \
	)

env_pip:
	if not exist $(VENV_DIR) ( \
		python -m venv $(VENV_DIR) && $(VENV_DIR)\Scripts\activate && pip install -r requirements.txt \
	)

all: env_conda
	@if not exist "$(CONDA_HOME)" ( \
		$(VENV_DIR)\Scripts\activate && \
		python $(SCRIPTS_DIR)/data_prep.py && \
		python $(SCRIPTS_DIR)/train_model.py && \
		python $(SCRIPTS_DIR)/evaluate_model.py; \
	) else ( \
		conda activate $(CONDA_ENV) && \
		python $(SCRIPTS_DIR)/data_prep.py && \
		python $(SCRIPTS_DIR)/train_model.py && \
		python $(SCRIPTS_DIR)/evaluate_model.py \
	)

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
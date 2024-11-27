DATA_DIR=data
SCRIPTS_DIR=scripts
MODELS_DIR=models
RESULTS_DIR=results

.PHONY: data train evaluate deploy clean

data:
	python $(SCRIPTS_DIR)/data_prep.py

train:
	python $(SCRIPTS_DIR)/train_model.py

evaluate:
	python $(SCRIPTS_DIR)/evaluate_model.py

deploy:
	python $(SCRIPTS_DIR)/deploy_model.py

clean:
	rm -rf $(DATA_DIR)/.csv $(MODELS_DIR)/.pkl $(RESULTS_DIR)/*.txt


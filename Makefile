.PHONY: setup clean

SHELL := /bin/bash
ENV_NAME := DoLQ

setup:
	@chmod +x install_env.sh
	@./install_env.sh

clean:
	@echo "Removing environment $(ENV_NAME)..."
	@conda remove -n $(ENV_NAME) --all -y
	@echo "Done."

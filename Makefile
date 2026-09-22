# Makefile — common workflows for fisheye-detection-training.
# Run `make` to list targets. Override variables inline, e.g.
#   make inspect PARTITION=validation LIMIT=0

PYTHON    ?= .venv/bin/python
MANIFEST  ?= data/manifests/person_detection_woodscape_version_1.yaml
PARTITION ?= train
LIMIT     ?= 300

# ROS setup scripts put /opt/ros/<distro>/.../site-packages on PYTHONPATH,
# which leaks Python 3.12 packages and pytest plugins into this Python 3.11
# environment. Keep this project isolated from it.
unexport PYTHONPATH
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

# Assumes the manifest filename matches identity.name inside it.
MANIFEST_NAME   := $(basename $(notdir $(MANIFEST)))
BUILD_DIRECTORY := data/build/$(MANIFEST_NAME)
BUILD_INFO      := $(BUILD_DIRECTORY)/build_info.json
BUILD_SOURCES   := tools/data/build_dataset.py src/data/source_readers.py engine/data_manifest.py

.DEFAULT_GOAL := help
.PHONY: help dataset dataset-rebuild inspect inspect-source test test-fast clean make toy-gate

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

dataset: $(BUILD_INFO) ## Build the dataset from MANIFEST, only if out of date

$(BUILD_INFO): $(MANIFEST) $(BUILD_SOURCES)
	$(PYTHON) -m tools.data.build_dataset --manifest $(MANIFEST) --force

dataset-rebuild: ## Force a full rebuild
	$(PYTHON) -m tools.data.build_dataset --manifest $(MANIFEST) --force

inspect: dataset ## Open the built PARTITION in FiftyOne
	$(PYTHON) -m tools.data.inspect_dataset --manifest $(MANIFEST) \
		--stage build --partition $(PARTITION) --limit $(LIMIT)

inspect-source: ## Open the raw source annotations in FiftyOne
	$(PYTHON) -m tools.data.inspect_dataset --manifest $(MANIFEST) \
		--stage source --limit $(LIMIT)

test: dataset ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-fast: ## Run tests that do not need the built dataset
	$(PYTHON) -m pytest tests/ -v --ignore=tests/test_dataset_build.py

toy-gate: ## Run the engine's toy training gate with output visible
	$(PYTHON) -m pytest tests/test_trainer.py -v -s
clean: ## Remove derived artifacts and caches
	rm -rf data/build .pytest_cache
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
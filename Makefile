.PHONY: all setup test lint format clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests

format:
	black src tests
	isort src tests

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

fetch-data:
	python scripts/fetch_data.py

build-corpus:
	python scripts/build_corpus.py

train-mt:
	python scripts/train_mt.py --train-data data/processed/mt_corpus_en_hi_train.jsonl

train-asr:
	python scripts/train_asr.py --train-data data/processed/asr_corpus_train.jsonl --val-data data/processed/asr_corpus_val.jsonl

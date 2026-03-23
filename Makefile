setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

train:
	python -m src.pipelines.training_pipeline

serve:
	uvicorn src.service.app:app --reload

ingest-debug:
	python -m src.data.ingest
.PHONY: install base imagens webcam test lint clean

install:
	uv sync

## Gera os embeddings das pessoas autorizadas
base:
	uv run python -m reconhecimento_facial.dataset

## Processa as fotos de teste e salva as imagens anotadas
imagens:
	uv run python -m reconhecimento_facial.modeling.predict imagens

## Reconhecimento em tempo real pela webcam (q para sair)
webcam:
	uv run python -m reconhecimento_facial.modeling.predict webcam

test:
	uv run pytest

lint:
	uv run ruff check .

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

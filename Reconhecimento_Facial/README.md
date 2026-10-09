# Reconhecimento_Facial

Controle de acesso por reconhecimento facial com **OpenCV** e **DeepFace**,
organizado no padrão **Cookiecutter Data Science (CCDS)**. Apenas scripts Python (sem notebooks).

- 🟩 Caixa **verde** + `acesso liberado` → rosto reconhecido como autorizado
- 🟥 Caixa **vermelha** + `acesso negado` → rosto desconhecido

## Estrutura

```
Reconhecimento_Facial/
├── data/
│   ├── raw/
│   │   ├── autorizados/   <- fotos de quem pode acessar (1 subpasta por pessoa)
│   │   └── teste/         <- fotos para testar o sistema
│   ├── interim/ external/
│   └── processed/resultados/   <- imagens geradas com as bounding boxes
├── models/                <- base de embeddings (base_autorizados.pkl)
├── reports/               <- resultados.csv
├── reconhecimento_facial/
│   ├── config.py          <- caminhos, modelo, limiar, cores
│   ├── dataset.py         <- gera a base das pessoas autorizadas
│   ├── features.py        <- detecção + embeddings (DeepFace) e regra de decisão
│   ├── plots.py           <- desenho das bounding boxes (OpenCV)
│   └── modeling/predict.py<- reconhecimento em fotos ou webcam
├── tests/  docs/  references/  notebooks/ (vazia, não usada)
├── Makefile  pyproject.toml  .env  .gitignore
```

## Como usar

1. Instale (Python 3.10–3.12): `uv sync`
2. Coloque as fotos autorizadas, uma pasta por pessoa:
   `data/raw/autorizados/Maria/foto1.jpg`, `.../Maria/foto2.jpg`, `.../Joao/foto1.jpg`
   (2–5 fotos nítidas por pessoa, rosto de frente, melhoram a precisão)
3. Coloque as fotos de teste em `data/raw/teste/` (inclua pessoas autorizadas **e** desconhecidas)
4. Gere a base: `make base`  (ou `uv run python -m reconhecimento_facial.dataset`)
5. Rode o reconhecimento: `make imagens`
   → imagens anotadas em `data/processed/resultados/` e relatório em `reports/resultados.csv`
6. Opcional, webcam: `make webcam` (q para sair)

> No Windows sem `make`, use direto os comandos `uv run python -m ...` do Makefile.

## Ajustes (arquivo `.env` ou `config.py`)

| Variável | Padrão | Observação |
|---|---|---|
| `FACE_MODEL` | `Facenet512` | também: `ArcFace`, `VGG-Face`, `Facenet` |
| `FACE_DETECTOR` | `opencv` | `retinaface` / `mtcnn` detectam melhor, porém mais lentos |
| `FACE_LIMIAR` | `0.30` | distância cosseno máxima para liberar; **menor = mais rígido**. Ajuste com suas fotos de teste. Ao trocar de modelo, recalibre. |

## Privacidade

Fotos de rosto são dados biométricos (LGPD). O `.gitignore` impede o versionamento das fotos e da base de embeddings.

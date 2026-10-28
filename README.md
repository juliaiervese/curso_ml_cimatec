# Repositório de Machine Learning e IA aplicada

Este repositório reúne projetos desenvolvidos ao longo da disciplina de Machine Learning, com foco em análise de dados, modelagem preditiva, deep learning, visão computacional e reconhecimento facial.

A organização está dividida em módulos independentes, cada um correspondente a uma etapa de aprendizagem e a um tema prático específico.

## Estrutura geral

```text
curso_ml_cimatec/
├── olist-ml/
├── deep_learning/
├── classificador_imagens/
├── Reconhecimento_Facial/
├── Multimodal_Generation/
├── README.md
└── .gitignore
```

## Pastas principais

### 1) olist-ml/
Projeto principal de análise e modelagem de dados do e-commerce Olist.

Funções principais:
- carregamento e preparação dos dados
- feature engineering
- split de treino/teste
- treinamento de modelos
- validação cruzada
- inferência
- explicabilidade com SHAP
- análise de qualidade dos dados
- exemplo de active learning

Estrutura típica:
- `module_olist/` — código da pipeline e módulos de modelagem
- `notebooks/` — notebooks de EDA, profiling e exemplos de estudo
- `data/` — dados brutos, intermediários e processados
- `models/` — modelos treinados e metadados
- `reports/` — relatórios, gráficos e artefatos de visualização
- `tests/` — testes de validação
- `README.md` — documentação específica do projeto

### 2) deep_learning/
Projeto de estudo em deep learning com foco em redes neurais e uso de frameworks como Keras e TensorFlow.

Funções principais:
- estrutura inicial para projeto de DL
- implementação de modelos com Keras
- implementação com TensorFlow
- notebooks de experimentação
- avaliação de performance

Estrutura típica:
- `module_deep_learning/` — módulos do projeto
- `notebooks/` — experimentos de Keras e TensorFlow
- `models/` — artefatos do modelo treinado
- `reports/` — gráficos e resultados
- `tests/` — validação do código

### 3) classificador_imagens/
Projeto de visão computacional para classificação de imagens.

Funções principais:
- estrutura de projeto para classificação de imagens
- carregamento e pré-processamento de imagens
- treinamento de rede para classificação
- geração de métricas e gráficos de desempenho
- estudo de resultados em conjunto de teste

Estrutura típica:
- `module_classificador_imagens/` — pipeline do classificador
- `data/` — imagens de entrada e dados processados
- `models/` — modelo treinado
- `reports/` — gráficos, métricas e histórico de treinamento
- `notebooks/` — experimentos e explorações

### 4) Reconhecimento_Facial/
Projeto de reconhecimento facial com aplicação prática de autenticação por rosto.

Funções principais:
- organização da base de imagens autorizadas e de teste
- geração de embeddings faciais
- detecção de rostos
- comparação com limiar de decisão
- geração de resultados e relatórios
- pipeline de reconhecimento em imagens ou webcam

Estrutura típica:
- `reconhecimento_facial/` — módulos do sistema
- `data/` — dados brutos e processados
- `models/` — base de embeddings
- `reports/` — CSVs e resultados de classificação
- `tests/` — testes do sistema
- `README.md` — instruções de uso

### 5) Multimodal_Generation/
Pasta adicional do repositório, representando geração multimodal ou projetos de extensão e experimentação.

Funções principais:
- armazenamento de projetos ou experimentos ligados a IA multimodal
- extensão do repositório para outros temas de inteligência artificial

Observação:
- a pasta pode conter projetos específicos, experimentos ou materiais complementares, conforme evolução do curso e das práticas de IA.

## Objetivo do repositório

O repositório funciona como uma coleção de projetos práticos que conectam:
- ciência de dados
- engenharia de features
- machine learning supervisionado
- deep learning
- computer vision
- reconhecimento facial
- explicabilidade de modelos

Essa organização permite acompanhar a evolução do aprendizado em diferentes frentes de IA aplicada, partindo de análise exploratória até soluções mais avançadas.

## Fluxo de aprendizado representado no repositório

O material foi organizado de forma progressiva:

1. Estrutura inicial e organização do projeto
2. Qualidade e limpeza dos dados
3. Análise exploratória
4. Engenharia de features
5. Treinamento de modelos
6. Validação cruzada
7. Inferência
8. Explicabilidade com SHAP
9. Active learning
10. Deep learning
11. Visão computacional
12. Reconhecimento facial

## Observações finais

- Cada pasta representa uma etapa ou área de estudo independente.
- O diretório principal serve como repositório de referência consolidada.
- O foco é demonstrar a aplicação prática de conceitos de IA e machine learning em projetos reais, com estrutura de código organizada e documentação complementar.

---

Este README foi criado para resumir a visão geral do repositório e orientar a navegação entre os projetos e módulos desenvolvidos ao longo da disciplina.

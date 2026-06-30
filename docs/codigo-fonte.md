# Documentação do Código-Fonte

Este documento descreve a organização do código do projeto `convnext-av1` e serve como referência rápida para manutenção, revisão e extensão do pipeline.

## Visão Geral

O projeto está organizado como um pacote Python instalável em `src/convnext_av1`. A interface principal é a CLI `convnext-av1`, definida em `src/convnext_av1/cli.py`, que conecta as etapas de conversão de dados, inspeção, divisão experimental, treino, avaliação, exportação e consolidação de benchmarks.

O fluxo principal é:

1. Instrumentar o libaom para gerar logs binários de decisões de partição.
2. Converter o `.bin` para dataset `.npz` validado e manifesto JSON.
3. Inspecionar amostras visualmente.
4. Criar splits por sequência.
5. Treinar o modelo ConvNeXt-AV1.
6. Avaliar acurácia, loss RD e métricas por nível/QP.
7. Exportar checkpoint para uso experimental no codec.
8. Consolidar resultados de benchmark.

## Módulos Python

`binary.py` define o contrato binário entre libaom e Python. Ele contém o tamanho fixo do registro, formato little-endian, validação dos campos e funções para empacotar/desempacotar amostras.

`converter.py` transforma logs binários em datasets `.npz`. Ele normaliza o bloco Y para `[0,1]`, preserva o `qindex` bruto, cria `qindex_norm`, registra contagens por nível/modo e escreve um manifesto JSON.

`data.py` carrega datasets convertidos, valida arrays obrigatórios, cria splits por sequência e expõe `AV1PartitionDataset` compatível com PyTorch.

`masks.py` implementa máscaras de validade geométrica para os 10 modos de partição AV1. Essas máscaras impedem que modos incompatíveis com bordas do frame sejam avaliados como candidatos válidos.

`model.py` implementa `ConvNeXtAV1`, com stem adaptado para canal Y, backbone ConvNeXt Tiny, ConvAdapters e três cabeças condicionadas ao QP para os níveis 64x64, 32x32 e 16x16.

`losses.py` contém a matriz de custo AV1 padrão e a `RDWeightedLoss`, usada para aproximar o custo esperado de erros de partição segundo uma interpretação RD-sensível.

`train.py` implementa o loop de treino, leitura de configuração YAML, controle de seed, checkpoint e relatório de treino.

`evaluate.py` carrega um checkpoint e calcula predições por nível, aplicando máscaras antes de selecionar o modo final.

`export.py` exporta checkpoints para TorchScript ou ONNX.

`inspect.py` salva amostras do dataset em PNG e gera relatório de sanidade.

`benchmark.py` consolida métricas de benchmark a partir de CSV para JSON.

`metrics.py` centraliza acurácia, matriz de confusão e agregações por nível/QP.

## CLI

A CLI expõe os seguintes comandos:

```bash
convnext-av1 convert
convnext-av1 inspect
convnext-av1 split
convnext-av1 train
convnext-av1 eval
convnext-av1 export
convnext-av1 benchmark
```

Cada comando delega a lógica para um módulo específico. A CLI deve permanecer fina: parsing de argumentos, chamada da função de domínio e impressão de JSON.

## Código do libaom

`tools/libaom/` contém os arquivos de referência para logging:

- `log_partition.h`: estrutura `av1_partition_sample_t` e macros condicionais.
- `log_partition.c`: abertura, escrita e fechamento do log binário.
- `README.md`: instruções de integração e build.

Todas as alterações no libaom devem permanecer protegidas por `LOG_PARTITION_DATA`, garantindo que builds sem essa flag mantenham o comportamento original.

## Testes

Os testes ficam em `tests/` e cobrem:

- contrato binário e tamanho de registro;
- conversão `.bin` para `.npz`;
- máscaras de partição;
- loss RD;
- shapes do modelo;
- comandos `convert` e `inspect`;
- split por sequência;
- consolidação de benchmark.

Comando recomendado:

```bash
.venv/bin/python -m pytest
```

## Convenções de Manutenção

- Novos textos devem ser escritos em PT-BR, conforme `AGENTS.md`.
- Identificadores de código podem permanecer em inglês para manter compatibilidade com Python, PyTorch e convenções do ecossistema.
- Alterações em dados experimentais devem produzir manifestos, não commits de arquivos pesados.
- Mudanças no modelo devem incluir pelo menos um teste de shape ou smoke test.
- Mudanças no formato binário devem atualizar `docs/data-format.md`, testes de parsing e a versão do schema.

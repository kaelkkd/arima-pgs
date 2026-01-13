# Pipeline de ARIMA com Grid Search Paralelo

Este repositório contém uma pipeline automatizada para a realização de estimativas do pre;o de fechamento de ativos financeiros utilizando o modelo ARIMA, com busca de hiperparâmetros otimizada via grid search paralelo.

## Descrição do Projeto

A implementação realiza um grid search paralelo para encontrar a ordem mais apropriada do modelo ARIMA. A ordem com menor AIC (Critério de Informação de Akaike) é escolhida como a definitiva, visando maior precisão nas estimativas.

## Instalação

### Usando um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate    # linux/macOS
venv\Scripts\activate       # windows

pip install -r requirements.txt
```

### Sem ambiente virtual

Instale diretamente as dependências:

```bash
pip install -r requirements.txt
```

## Como usar

O arquivo `main.py` possui presets de uso que podem ser executados diretamente ou customizados de acordo a preferência do usuário:

```bash
python main.py
```

Ou usando um preset específico:

```bash
python main.py --preset medium
```

Também é possível customizar a execução diretamente pelo terminal:

```bash
python main.py --ticker ITUB4.SA --start-date 2023-01-01 --end-date 2024-01-01 --p-upper 5 --q-upper 5 --chunk-size 8
```
Recomenda-se o uso de limites abaixo de 7 para os parâmetros *p* e *q* devido a natureza do modelo. Para os chunks recomenda-se valores entre 5 e 20 para um melhor aproveitamento do paralelismo.

### Opções de personalização

* `--ticker`: nome do ativo (deve corresponder ao usado no Yahoo! Finance)
* `--start-date` e `--end-date`: período de análise (formato AAAA-MM-DD)
* `--p-upper` e `--q-upper`: limite superior para os parâmetros *p* e *q* do ARIMA no grid search
* `--chunk-size`: número de ordens distribuídas para cada job
* `--no-capture`: exibe apenas o output retornado no terminal
* `--no-log`: não salva o output em arquivo externo
* `--log-file`: define caminho personalizado para salvar o arquivo de log
* `--show-plot`: mostra o grafico de preços reais vs. estimativas
* `--save-image /path/`: define o caminho para salvar o gráfico (deve ser usado junto do anterior caso deseja exibir e salvar o gráfico)
Caso o usuário deseje imagens com uma melhor resolução, deve-se utilizar o formato PDF. O usuário também pode customizar o gráfico utilizando TeX. Neste caso, recomenda-se a instalação de uma distribuição TeX como [MiKTeX](https://miktex.org/). 


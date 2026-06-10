# Estratégias de Ranking

Este arquivo resume as estratégias de ranking usadas no sweep do projeto. Elas recebem as listas de hits produzidas pelas janelas de consulta e transformam isso em uma lista final de espécies ordenadas por score.

## Visão Geral

O objetivo dessas estratégias não é mudar o embedding nem a segmentação. Elas atuam apenas na etapa final de ordenação, então servem para responder perguntas como:

- qual espécie deve ficar em primeiro lugar;
- quais espécies devem aparecer logo depois;
- como combinar evidências vindas de vários segmentos do mesmo áudio.

## Estratégias

### `segment`

Mantém, para cada espécie, o maior score encontrado em qualquer segmento da consulta. É a forma mais direta de usar os resultados.

Quando usar: quando a prioridade é acertar o top-1 e você quer uma regra simples e agressiva.

### `max`

É equivalente a `segment` nesta implementação. A diferença é mais de nomenclatura do que de comportamento.

Quando usar: quando você quer deixar explícito que a regra é baseada no maior score observado.

### `mean`

Calcula a média de todos os scores de uma espécie ao longo das janelas de consulta. Isso suaviza picos isolados e valoriza concordância entre segmentos.

Quando usar: quando você quer uma ordenação mais estável e menos sensível a um único hit muito forte.

### `topk_mean`

Seleciona os melhores scores de cada espécie e faz a média apenas desses valores. É um meio-termo entre `max` e `mean`.

Quando usar: quando você quer preservar os melhores sinais sem deixar um único valor dominar a decisão.

### `hit`

Conta em quantas janelas de consulta a espécie apareceu. Aqui o score final é a frequência de aparição, não a intensidade do match.

Quando usar: quando a recorrência importa mais do que a força absoluta do score.

### `median`

Usa a mediana dos scores da espécie. Essa estratégia tende a ser robusta a outliers e ruídos extremos.

Quando usar: quando há segmentos barulhentos e você quer reduzir o impacto de valores atípicos.

### `threshold`

Mantém apenas os hits acima de um limiar `tau` e depois calcula a média desses valores. Qualquer espécie sem evidência suficiente pode desaparecer do ranking.

Quando usar: quando você quer filtrar evidências fracas com mais rigor.

### `weighted_topk`

Pega os melhores scores de cada espécie e aplica pesos maiores aos primeiros, com decaimento linear. Isso dá mais importância aos melhores sinais sem ignorar completamente os demais.

Quando usar: quando os melhores hits costumam ser bem confiáveis, mas você ainda quer alguma suavização.

### `softmax`

Aplica softmax dentro de cada lista de consulta e soma os pesos por espécie. Os hits mais fortes recebem maior concentração de peso.

Quando usar: quando você quer uma fusão suave, mas ainda com foco claro nos maiores scores.

### `rrf`

Reciprocal Rank Fusion. Em vez de usar o score bruto, combina as listas pela posição de cada espécie no ranking de cada janela.

Quando usar: quando várias janelas tendem a concordar sobre as mesmas espécies e a posição na lista é mais importante do que o score exato.

### `borda`

Usa a contagem de pontos baseada na posição de cada espécie em cada lista. Espécies melhor colocadas recebem mais pontos.

Quando usar: quando você quer uma fusão clássica, simples e estável de múltiplas listas.

### `attention`

É parecida com `softmax`, mas pode dar mais peso para consultas mais confiantes usando o pico de similaridade como fator adicional.

Quando usar: quando algumas janelas do áudio são muito mais informativas do que outras.

### `taxonomy_boost`

Parte do score base e adiciona bônus para espécies taxonomicamente próximas ao melhor candidato. A implementação favorece espécies do mesmo gênero e também nomes comuns com termos parecidos.

Quando usar: quando o objetivo não é só acertar a espécie exata, mas também trazer espécies biologicamente próximas no topo da lista.

## Leitura prática

Se o foco for apenas acertar a espécie principal, `segment` e `max` costumam ser suficientes. Se a meta for melhorar as posições seguintes da lista, normalmente vale testar `rrf`, `attention`, `topk_mean` e `taxonomy_boost`.

Em geral:

- `segment` e `max` priorizam o melhor sinal individual;
- `mean`, `median` e `threshold` suavizam ruído;
- `topk_mean`, `weighted_topk`, `softmax`, `rrf` e `borda` fazem fusão de múltiplas evidências;
- `attention` e `taxonomy_boost` tentam produzir listas mais úteis do ponto de vista biológico.

## Relação com o projeto

No sweep de ranking, o embedding e a segmentação permanecem fixos. A única coisa que muda é o `ranking.type` no YAML. Isso permite comparar estratégias de forma justa.

## Comandos úteis

Para rodar o sweep principal:

```bash
./venv/bin/python experiments/ranking_sweep_all.py
```

Para testar configs BirdNET v3.0:

```bash
./venv/bin/python experiments/ranking_sweep_all.py --configs \
  configs/strategy1_segments_birdnet_v3.yaml \
  configs/strategy2_super_embedding_birdnet_v3.yaml
```

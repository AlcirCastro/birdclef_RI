# Sweeps de Ranking

Este experimento avalia, de forma comparativa, várias estratégias de ranking sobre a mesma base de busca. A ideia é medir não só a qualidade da primeira posição, mas também como a lista inteira se comporta quando queremos recuperar espécies próximas taxonomicamente, por exemplo fazer com que uma consulta de "sapo" traga outros sapos ou animais muito parecidos.

## Objetivo

O `ranking_sweep_all.py` existe para responder a uma pergunta prática: qual estratégia produz o melhor reranking depois que o embedding já encontrou candidatos plausíveis?

Ele foi pensado para cenários em que o top-1 já é bom, mas as posições seguintes ainda estão fracas. Nessa situação, um reranker melhor pode melhorar a consistência da lista e trazer espécies próximas em vez de candidatos aleatórios.

## O que ele executa

Por padrão, o script roda os dois configs Torch principais:

- `configs/strategy1_segments_torch.yaml`
- `configs/strategy2_super_embedding_torch.yaml`

Em cada config ele troca apenas o `ranking.type`, mantendo o restante do pipeline fixo.

As estratégias avaliadas por padrão são:

- `segment`
- `mean`
- `max`
- `topk_mean`
- `hit`
- `median`
- `threshold`
- `weighted_topk`
- `softmax`
- `rrf`
- `borda`
- `attention`
- `taxonomy_boost`

O `taxonomy_boost` é o reranker novo que tenta favorecer espécies do mesmo gênero ou com pistas parecidas no nome comum.

## Explicação de cada estratégia

- `segment`: usa a maior evidência individual de qualquer segmento. É a opção mais direta e costuma favorecer muito o top-1, mas pode deixar a cauda da lista fraca.
- `mean`: faz a média dos scores de todos os hits por espécie. Tende a ser mais estável que `max` porque recompensa concordância entre vários trechos.
- `max`: mantém o maior score visto para cada espécie. Na prática é parecido com `segment` e costuma ser forte para a primeira posição, mas pouco suave para itens seguintes.
- `topk_mean`: média dos melhores `k` scores por espécie. É um meio-termo entre `max` e `mean`; tenta preservar o sinal forte sem deixar um único score dominar tudo.
- `hit`: conta em quantas janelas de consulta a espécie apareceu. É útil quando você quer valorizar recorrência, mas pode ignorar intensidade de score.
- `median`: usa a mediana dos scores por espécie. É mais robusto a outliers e pode ajudar quando alguns segmentos da consulta são barulhentos.
- `threshold`: considera apenas hits acima de um limiar `tau`. Serve para descartar evidências fracas, mas pode zerar espécies que aparecem pouco.
- `weighted_topk`: dá peso maior aos scores mais altos e menor aos demais. Costuma melhorar a ordenação quando há poucos hits muito bons.
- `softmax`: transforma scores em pesos suaves por consulta. É uma forma de concentrar a atenção nos hits mais fortes sem perder completamente os outros.
- `rrf`: Reciprocal Rank Fusion. É forte quando várias janelas concordam em espécies parecidas, porque combina posição de ranking e não só score bruto.
- `borda`: soma pontos pela posição em cada lista. É uma fusão clássica para reduzir a influência de um único score extremo.
- `attention`: versão mais flexível da softmax, com peso por confiança da consulta. É útil quando algumas janelas do áudio são muito mais informativas que outras.
- `taxonomy_boost`: parte do score base e adiciona um bônus para espécies taxonomicamente próximas ao top candidato. É a melhor aposta quando a intenção é sair de "mesma classe acústica" e ir para "mesma família/gênero".

### Leitura prática dessas diferenças

Se o problema é só acertar a espécie principal, `max` e `segment` podem ser suficientes. Se o problema é melhorar as posições 2, 3, 4... então normalmente vale testar `rrf`, `attention`, `topk_mean` e `taxonomy_boost`. Esses quatro tendem a produzir listas mais “humanamente plausíveis”, com espécies próximas aparecendo logo depois do topo.

## Como funciona

O script carrega um YAML, substitui apenas a configuração de ranking e executa o `ExperimentRunner` normal do framework. Depois coleta as métricas do bloco `clean` e salva um resumo em:

- `results/ranking_sweep_all/comparison.json`
- `results/ranking_sweep_all/comparison.md`

## Como executar

Use sempre o Python do ambiente virtual:

```bash
./venv/bin/python experiments/ranking_sweep_all.py
```

Se quiser avaliar configs diferentes:

```bash
./venv/bin/python experiments/ranking_sweep_all.py --configs \
  configs/strategy1_segments_torch.yaml \
  configs/strategy2_super_embedding_torch.yaml
```

Se quiser testar só alguns rankers:

```bash
./venv/bin/python experiments/ranking_sweep_all.py --rankings rrf attention taxonomy_boost
```

## Como ler os resultados

O arquivo `comparison.md` traz uma tabela com:

- `MAP`
- `MRR`
- `P@1`
- `P@5`
- `R@1`
- `R@5`
- `R@10`
- `nDCG`
- latência média

Se o seu problema é “top-1 bom, resto ruim”, os campos mais úteis são:

- `P@5` e `R@5`, para ver se a lista está ficando mais útil além da primeira posição
- `nDCG`, para medir se os itens mais altos estão mais bem ordenados
- `taxonomy_boost`, para verificar se o reranking taxonômico melhora a cauda da lista

## Quando usar cada estratégia

- `rrf`: bom quando várias janelas concordam sobre a mesma espécie.
- `attention`: bom quando algumas janelas da consulta são muito mais informativas que outras.
- `topk_mean`: bom quando você quer suavizar ruído sem perder muito sinal forte.
- `taxonomy_boost`: bom quando você quer trazer espécies da mesma família/gênero, ou nomes comuns parecidos.

## Observações

- O sweep não muda o embedding nem a segmentação.
- Ele avalia apenas o efeito do ranking final.
- O backend Torch é mantido, então o experimento continua compatível com a versão multi-GPU do embedder.
- Se quiser um ranking mais biológico, o próximo passo natural é enriquecer a taxonomia com uma tabela de família/gênero por espécie, em vez de usar só o nome científico e o nome comum.
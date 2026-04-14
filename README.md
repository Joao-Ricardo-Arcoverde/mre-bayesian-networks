# Most Relevant Explanation in Bayesian Networks

Implementação em Python do paper de Yuan, Lim e Lu (2011), desenvolvida como parte de uma pesquisa de mestrado. O objetivo foi traduzir fielmente os algoritmos e medidas descritos no artigo em código executável, reproduzindo os resultados das tabelas originais e tornando o raciocínio do paper acessível de forma interativa.

> Yuan, C., Lim, H., & Lu, T.-C. (2011). **Most Relevant Explanation in Bayesian Networks**.
> *Journal of Artificial Intelligence Research*, 42, 309–352.
> [https://arxiv.org/abs/1401.3893](https://arxiv.org/abs/1401.3893)

---

## O que é o MRE

Redes Bayesianas são boas em calcular probabilidades, mas uma pergunta diferente é: **por que** um determinado evento aconteceu? Isso é o problema da explicação. As abordagens clássicas (MAP e MPE) têm limitações bem conhecidas — o MAP tende a especificar demais e o MPE frequentemente retorna explicações redundantes.

O MRE (*Most Relevant Explanation*) resolve isso usando o **Generalized Bayes Factor (GBF)** para medir o quanto cada instanciação parcial das variáveis é *relevante* para explicar a evidência observada. A ideia central é encontrar explicações que sejam ao mesmo tempo informativas e parcimoniosas — sem variáveis desnecessárias, sem sub-especificação.

---

## Estrutura do repositório

```
.
├── mre.py                             # biblioteca com todas as medidas e algoritmos
├── MRE_em_Redes_Bayesianas_light.ipynb   # notebook principal (desenvolvimento completo)
├── Demonstrando_mre.ipynb             # notebook de demonstração usando a biblioteca
└── requirements.txt
```

---

## O notebook principal

O `MRE_em_Redes_Bayesianas_light.ipynb` percorre o paper de ponta a ponta, com código e explicação lado a lado. As seções seguem a progressão lógica do artigo:

**Contexto e motivação** — começa com uma breve história dos sistemas especialistas (MYCIN) e de como as Redes Bayesianas surgiram como alternativa mais principiada. A rede Ásia é usada como exemplo introdutório, explorando três cenários clínicos distintos com propagação de crença.

**Explanation in Bayesian Networks** — apresenta o problema da explicação, discute o *underspecification* (BUR / Belief Update Rate) e o *overspecification* (MAP/MPE), e introduz as redes da Viagem (Shimony, 1993) e do Circuito Elétrico (Poole & Provan, 1991) como casos motivadores.

**Most Relevant Explanation** — derivação completa do GBF (Eq. 4), prova dos Corolários 1, 2 e 3, o CBF (*Conditional Bayes Factor*) e o fenômeno de *explaining away*. Em seguida, as definições de dominância forte e fraca, o conceito de explicação minimal, e o algoritmo K-MRE com pre-pruning e post-pruning.

**Estudos de caso** — reprodução das tabelas do paper nas quatro redes:
- **Viagem** (1 estado e multi-estado): MRE vs. MAP, com e sem *explaining away*
- **Circuito Elétrico**: K-MRE reproduz a Tabela 2 do paper
- **Academe**: desempenho acadêmico com 4 variáveis-alvo, Tabela 6
- **Ásia**: K-MRE com dominância forte e fraca, Tabela 7

---

## A biblioteca `mre.py`

Contém todas as medidas e algoritmos do paper como funções independentes, prontas para usar em qualquer rede pgmpy.

### Medidas fundamentais

| Função | Descrição |
|---|---|
| `gbf(post, prior)` | Generalized Bayes Factor (Eq. 4) |
| `bur(post, prior)` | Belief Update Ratio (Eq. 1) |
| `cbf(post_cond, prior_cond)` | Conditional Bayes Factor (Def. 4) |
| `jeffreys(valor)` | Classificação qualitativa pela escala de Jeffreys (Tabela 1) |

### Explicação

| Função | Descrição |
|---|---|
| `todas_instanciacoes(...)` | Calcula o GBF de todas as instanciações parciais |
| `mre(...)` | Retorna a melhor explicação (máximo GBF) |
| `kmre(...)` | K melhores explicações com dominância |
| `kmap(...)` | K melhores explicações pelo MAP |
| `ksimp(...)` | Versão simplificada com threshold |

### Dominância e minimalidade

| Função | Descrição |
|---|---|
| `domina_forte(a, b)` | Dominância forte (Def. 6) |
| `domina_fraca(a, b)` | Dominância fraca (Def. 7) |
| `is_minimal(exp, todas)` | Verifica se uma explicação é minimal (Def. 8) |

### Utilitários

| Função | Descrição |
|---|---|
| `posteriors(infer, alvos, evidencia)` | Posteriors de todas as variáveis-alvo |
| `cbf_condicional(...)` | CBF dado que outra variável já está na explicação |
| `check_d_separado(modelo, inicio, fim, observados)` | Teste de d-separação |
| `get_example_model(nome)` | Retorna Circuit, Vacation ou Academe prontos para uso |

---

## Instalação

```bash
# Ubuntu/Debian — necessário para pygraphviz
sudo apt-get install -y graphviz graphviz-dev

pip install pgmpy pygraphviz matplotlib pandas numpy jupyter
```

No Google Colab, os dois primeiros comandos podem ser rodados diretamente na célula inicial do notebook (já incluídos).

---

## Exemplo rápido

```python
from mre import get_example_model, kmre

# Circuito elétrico com evidência: input=current, output=current
ex = get_example_model('circuit')
resultados = kmre(
    ex['infer'],
    ex['alvos'],
    ex['evidencias']['corrente'],
    ex['estados'],
    ex['priors'],
    k=3
)

for r in resultados:
    print(r['label'], f"  GBF = {r['gbf']:.2f}")
```

Saída esperada (reproduz a Tabela 2 do paper):
```
B=defective, C=defective   GBF = 42.62
A=defective                GBF = 39.44
B=defective, D=defective   GBF = 35.88
```

---

## Dependências

- Python 3.9+
- [pgmpy](https://pgmpy.org/)
- numpy, pandas, matplotlib
- pygraphviz (para visualização das redes)

---

## Contexto acadêmico

Este repositório foi desenvolvido no contexto de uma dissertação de mestrado, com orientação do Prof. Carlos Maciel. O objetivo central foi tornar o paper de Yuan et al. (2011) reproduzível e didático, servindo de material de apoio tanto para estudo do MRE quanto para pesquisas que queiram usar o GBF como critério de explicação em suas próprias redes.

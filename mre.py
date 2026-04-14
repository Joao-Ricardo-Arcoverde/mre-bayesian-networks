"""
mre_lib.py — Most Relevant Explanation in Bayesian Networks
============================================================
Implementacao das medidas e metodos do paper:

    Yuan, C., Lim, H., & Lu, T.-C. (2011).
    Most Relevant Explanation in Bayesian Networks.
    Journal of Artificial Intelligence Research, 42, 309-352.

Compativel com redes pgmpy (BayesianNetwork / DiscreteBayesianNetwork).

Medidas:
    gbf(post, prior)             — Generalized Bayes Factor (Eq. 4)
    bur(post, prior)             — Belief Update Ratio (Eq. 1)
    cbf(post_cond, prior_cond)   — Conditional Bayes Factor (Def. 4)
    jeffreys(valor)              — Classificacao qualitativa (Tabela 1)

Explicacao:
    todas_instanciacoes(infer, alvos, evidencia, estados, priors)
    mre(infer, alvos, evidencia, estados, priors)
    kmre(infer, alvos, evidencia, estados, priors, k)
    kmap(infer, alvos, evidencia, k)
    ksimp(infer, alvos, evidencia, estados, priors, k, threshold)

Dominancia:
    domina_forte(a, b)
    domina_fraca(a, b)
    is_minimal(explicacao, todas)

Utilitarios de rede:
    posteriors(infer, alvos, evidencia)
    cbf_condicional(infer, alvo_spec, condicao_spec, evidencia, prior_alvo)
    check_d_separado(modelo, inicio, fim, observados)
"""

import numpy as np
from itertools import combinations, product


# métricas fundamentais

def gbf(post: float, prior: float) -> float:
    """
    Generalized Bayes Factor (Equacao 4 do paper).

    GBF(x;e) = P(x|e)(1-P(x)) / (P(x)(1-P(x|e)))

    Mede o quanto a evidencia e muda as odds da hipotese x.
    GBF > 1  : evidencia suporta x
    GBF = 1  : evidencia e irrelevante para x
    GBF < 1  : evidencia contradiz x

    Parametros
    ----------
    post  : P(x|e)  — probabilidade posterior de x dado a evidencia
    prior : P(x)    — probabilidade a priori de x

    Retorna
    -------
    float : valor do GBF; 0.0 para casos degenerados
    """
    if prior <= 0 or prior >= 1:
        return 0.0
    if post <= 0:
        return 0.0
    if post >= 1.0:
        return float('inf')
    return (post * (1.0 - prior)) / (prior * (1.0 - post))


def bur(post: float, prior: float) -> float:
    """
    Belief Update Ratio (Equacao 1 do paper).

    r(x;e) = P(x|e) / P(x)

    Equivalente a P(e|x) / P(e). Proporcional ao likelihood.
    Nao penaliza conjuncao irrelevante — limitacao citada no paper.

    Parametros
    ----------
    post  : P(x|e)
    prior : P(x)

    Retorna
    -------
    float
    """
    if prior <= 0:
        return 0.0
    return post / prior


def cbf(post_cond: float, prior_cond: float) -> float:
    """
    Conditional Bayes Factor (Definicao 4 do paper).

    CBF(y; e | x) = GBF de y dado que x ja esta na explicacao.

    Usado para medir o explaining-away: se CBF(y|x) < 1/r(x;e),
    adicionar y a uma explicacao que ja contem x reduz o GBF total.

    Parametros
    ----------
    post_cond  : P(y|e, x)  — posterior de y dado evidencia E e explicacao x
    prior_cond : P(y|x)     — prior de y condicionado em x
                              (= P(y) se y e independente de x a priori)

    Retorna
    -------
    float : valor do CBF; equivalente a gbf(post_cond, prior_cond)
    """
    return gbf(post_cond, prior_cond)


def jeffreys(valor: float) -> str:
    """
    Classificacao qualitativa de um GBF/CBF segundo a escala de Jeffreys
    (Tabela 1 do paper).

    Parametros
    ----------
    valor : float — valor do GBF ou CBF

    Retorna
    -------
    str : classificacao da forca da evidencia
    """
    escala = [
        (0,    1,          "Negativa"),
        (1,    3,          "Quase irrelevante"),
        (3,    10,         "Substancial"),
        (10,   30,         "Forte"),
        (30,   100,        "Muito forte"),
        (100,  float('inf'), "Decisiva"),
    ]
    for low, high, label in escala:
        if low <= valor < high:
            return label
    return "Indefinida"


# dominância entre explicações

def _is_subset(spec_a: dict, spec_b: dict) -> bool:
    """True se spec_a e subconjunto ESTRITO de spec_b."""
    return set(spec_a.items()) < set(spec_b.items())


def domina_forte(a: dict, b: dict) -> bool:
    """
    Dominancia forte (Definicao 6 do paper).

    a domina fortemente b se:
        - a e subconjunto estrito de b  (a eh mais conciso)
        - GBF(a) >= GBF(b)              (a eh igualmente bom ou melhor)

    Parametros
    ----------
    a, b : dict com chaves 'spec' (dict) e 'gbf' (float)

    Retorna
    -------
    bool
    """
    return _is_subset(a['spec'], b['spec']) and a['gbf'] >= b['gbf']


def domina_fraca(a: dict, b: dict) -> bool:
    """
    Dominancia fraca (Definicao 7 do paper).

    a domina fracamente b se:
        - a e superconjunto estrito de b  (a eh mais complexo)
        - GBF(a) > GBF(b)                (mas a eh estritamente melhor)

    Parametros
    ----------
    a, b : dict com chaves 'spec' (dict) e 'gbf' (float)

    Retorna
    -------
    bool
    """
    return _is_subset(b['spec'], a['spec']) and a['gbf'] > b['gbf']


def is_minimal(explicacao: dict, todas: list) -> bool:
    """
    Verifica se uma explicacao e minimal (Definicao 8 do paper).

    Uma explicacao e minimal se nao e dominada — nem forte nem fracamente —
    por nenhuma outra explicacao da lista.

    Parametros
    ----------
    explicacao : dict com 'spec' e 'gbf'
    todas      : lista de dicts com 'spec' e 'gbf'

    Retorna
    -------
    bool
    """
    return not any(
        (domina_forte(outra, explicacao) or domina_fraca(outra, explicacao))
        for outra in todas
        if outra is not explicacao
    )


# instâncias parciais

def todas_instanciacoes(
    infer,
    alvos: list,
    evidencia: dict,
    estados: dict,
    priors: dict,
) -> list:
    """
    Calcula o GBF de TODAS as instanciacoes parciais dos alvos.

    Para k variaveis-alvo com n_i estados cada, gera todas as
    2^k - 1 combinacoes nao-vazias de subconjuntos e todos os seus estados.

    Parametros
    ----------
    infer     : VariableElimination do pgmpy
    alvos     : lista de nomes das variaveis-alvo (ex: ['tub','lung','bronc'])
    evidencia : dict de evidencias observadas (ex: {'dysp':'yes'})
    estados   : dict {variavel: [lista de estados]}
                ex: {'tub':['yes','no'], 'lung':['yes','no']}
    priors    : dict {variavel: {estado: probabilidade}}
                ex: {'tub':{'yes':0.01,'no':0.99}}

    Retorna
    -------
    list de dicts, cada um com:
        'label'  : str — descricao legivel da instanciacao
        'spec'   : dict {variavel: estado}
        'gbf'    : float — GBF da instanciacao
        'p_post' : float — P(spec | evidencia)
        'p_prior': float — P(spec) a priori
    """
    q_conjunta = infer.query(alvos, evidence=evidencia, show_progress=False)
    resultados = []

    for r in range(1, len(alvos) + 1):
        for subset in combinations(alvos, r):
            for combo in product(*[estados[v] for v in subset]):
                spec = dict(zip(subset, combo))

                # Marginalizar sobre variaveis nao especificadas
                vars_fora = [v for v in alvos if v not in subset]
                if vars_fora:
                    q_marg = q_conjunta.marginalize(vars_fora, inplace=False)
                    p_post = q_marg.get_value(**spec)
                else:
                    p_post = q_conjunta.get_value(**spec)

                p_prior = float(np.prod([priors[v][s] for v, s in spec.items()]))
                g = gbf(p_post, p_prior)
                label = ', '.join(f'{v}={s}' for v, s in sorted(spec.items()))

                resultados.append({
                    'label':   label,
                    'spec':    spec,
                    'gbf':     g,
                    'p_post':  p_post,
                    'p_prior': p_prior,
                })

    return sorted(resultados, key=lambda x: -x['gbf'])


# MRE e KMRE

def mre(
    infer,
    alvos: list,
    evidencia: dict,
    estados: dict,
    priors: dict,
) -> dict:
    """
    Most Relevant Explanation (Definicao 5 do paper).

    Retorna a instanciacao parcial dos alvos que maximiza o GBF.

    MRE(M; e) = argmax_{x, X subset M} GBF(x; e)

    Parametros
    ----------
    infer, alvos, evidencia, estados, priors : ver todas_instanciacoes()

    Retorna
    -------
    dict com 'label', 'spec', 'gbf', 'p_post', 'p_prior'
    """
    todas = todas_instanciacoes(infer, alvos, evidencia, estados, priors)
    return todas[0]

def kmre(
    infer,
    alvos: list,
    evidencia: dict,
    estados: dict,
    priors: dict,
    k: int = 3,
) -> list:
    """
    K-MRE: Top-K Most Relevant Explanations (Secao 4.6 do paper).

    Encontra as k explicacoes minimais com maior GBF.
    Explicacoes dominadas (forte ou fracamente) sao removidas para garantir
    diversidade e representatividade.

    Parametros
    ----------
    infer, alvos, evidencia, estados, priors : ver todas_instanciacoes()
    k : int — numero de explicacoes desejadas (default: 3)

    Retorna
    -------
    list de dicts, cada um com 'label', 'spec', 'gbf', 'p_post', 'p_prior'
    ordenados por GBF decrescente
    """
    todas = todas_instanciacoes(infer, alvos, evidencia, estados, priors)
    minimais = [r for r in todas if is_minimal(r, todas)]
    return sorted(minimais, key=lambda x: -x['gbf'])[:k]


# K-MAP  e K-SIMP

def kmap(
    infer,
    alvos: list,
    evidencia: dict,
    estados: dict,
    k: int = 3,
) -> list:
    """
    K-MAP: Top-K configuracoes completas de maxima probabilidade posterior.

    Diferente do MRE, o MAP sempre especifica TODOS os alvos — o que causa
    overspecification. Incluido para comparacao direta com K-MRE.

    Parametros
    ----------
    infer     : VariableElimination do pgmpy
    alvos     : lista de variaveis-alvo
    evidencia : dict de evidencias
    estados   : dict {variavel: [lista de estados]}
    k         : numero de configuracoes desejadas

    Retorna
    -------
    list de dicts com 'label', 'spec', 'p_post'
    """
    q_conjunta = infer.query(alvos, evidence=evidencia, show_progress=False)
    resultados = []

    for combo in product(*[estados[v] for v in alvos]):
        spec = dict(zip(alvos, combo))
        p_post = q_conjunta.get_value(**spec)
        label = ', '.join(f'{v}={s}' for v, s in sorted(spec.items()))
        resultados.append({'label': label, 'spec': spec, 'p_post': p_post})

    return sorted(resultados, key=lambda x: -x['p_post'])[:k]


def ksimp(
    infer,
    alvos: list,
    evidencia: dict,
    estados: dict,
    priors: dict,
    k: int = 3,
    threshold: float = 0.05,
) -> list:
    """
    K-MAP Simplification (de Campos et al., 2001) — Secao 3.2.1 do paper.

    Parte das top-K configuracoes MAP e remove variaveis cuja remocao
    nao reduz a likelihood da evidencia alem do threshold.

    Parametros
    ----------
    infer, alvos, evidencia, estados, priors : ver todas_instanciacoes()
    k         : numero de solucoes MAP de partida
    threshold : reducao maxima de likelihood permitida para remocao (default: 0.05)

    Retorna
    -------
    list de dicts com 'label', 'spec', 'likelihood'
    (dedupados e ordenados por likelihood decrescente)
    """
    configs_map = kmap(infer, alvos, evidencia, estados, k=k * 3)
    vistas = {}

    for cfg in configs_map:
        spec_atual = dict(cfg['spec'])

        melhorou = True
        while melhorou:
            melhorou = False
            for var in list(spec_atual.keys()):
                spec_sem = {v: s for v, s in spec_atual.items() if v != var}
                if not spec_sem:
                    continue

                ev_com  = {**evidencia, **spec_atual}
                ev_sem  = {**evidencia, **spec_sem}

                q_conjunta = infer.query(alvos, evidence=evidencia, show_progress=False)

                vars_fora_com = [v for v in alvos if v not in spec_atual]
                if vars_fora_com:
                    q_com = q_conjunta.marginalize(vars_fora_com, inplace=False)
                    like_com = q_com.get_value(**spec_atual)
                else:
                    like_com = q_conjunta.get_value(**spec_atual)

                vars_fora_sem = [v for v in alvos if v not in spec_sem]
                if vars_fora_sem:
                    q_sem = q_conjunta.marginalize(vars_fora_sem, inplace=False)
                    like_sem = q_sem.get_value(**spec_sem)
                else:
                    like_sem = q_conjunta.get_value(**spec_sem)

                delta = like_com - like_sem
                if delta <= threshold * like_com:
                    spec_atual = spec_sem
                    melhorou = True
                    break

        label = ', '.join(f'{v}={s}' for v, s in sorted(spec_atual.items()))
        q_conjunta = infer.query(alvos, evidence=evidencia, show_progress=False)
        vars_fora = [v for v in alvos if v not in spec_atual]
        if vars_fora:
            q_f = q_conjunta.marginalize(vars_fora, inplace=False)
            like_final = q_f.get_value(**spec_atual)
        else:
            like_final = q_conjunta.get_value(**spec_atual)

        if label not in vistas or like_final > vistas[label]['likelihood']:
            vistas[label] = {'label': label, 'spec': spec_atual, 'likelihood': like_final}

    return sorted(vistas.values(), key=lambda x: -x['likelihood'])[:k]


# utilitários

def posteriors(
    infer,
    alvos: list,
    evidencia: dict,
) -> dict:
    """
    Calcula as posteriors marginais de cada variavel-alvo (Belief Updating).

    Retorna um dict {variavel: {estado: probabilidade}} com todas as
    distribuicoes a posteriori marginalizadas individualmente.

    Parametros
    ----------
    infer     : VariableElimination do pgmpy
    alvos     : lista de variaveis
    evidencia : dict de evidencias

    Retorna
    -------
    dict {variavel: DiscreteFactor do pgmpy}
    """
    return {
        v: infer.query([v], evidence=evidencia, show_progress=False)
        for v in alvos
    }


def cbf_condicional(
    infer,
    alvo_spec: dict,
    condicao_spec: dict,
    evidencia: dict,
    prior_alvo: float,
) -> float:
    """
    Calcula o CBF de uma variavel condicionada em uma explicacao existente.

    CBF(y; e | x) mede o quanto y acrescenta a uma explicacao que ja contem x.

    Se CBF <= 1/r(x;e), adicionar y nao melhora a explicacao — pelo Teorema 2.

    Parametros
    ----------
    infer          : VariableElimination do pgmpy
    alvo_spec      : dict {variavel: estado} — a variavel sendo avaliada
                     ex: {'A': 'defective'}
    condicao_spec  : dict {variavel: estado} — explicacao ja existente
                     ex: {'B': 'defective', 'C': 'defective'}
    evidencia      : dict de evidencias do problema
    prior_alvo     : P(y) — prior marginal do alvo (independente da condicao)

    Retorna
    -------
    float : valor do CBF; equivalente a gbf(P(y|e,x), P(y|x))
    """
    ev_cond = {**evidencia, **condicao_spec}
    q = infer.query(list(alvo_spec.keys()), evidence=ev_cond, show_progress=False)
    p_post_cond = q.get_value(**alvo_spec)
    return cbf(p_post_cond, prior_alvo)


def check_d_separado(
    modelo,
    inicio: str,
    fim: str,
    observados: list,
) -> bool:
    """
    Verifica d-separacao entre dois nos dado um conjunto de observados.

    Usa active_trail_nodes do pgmpy para verificar se ha caminho ativo.
    Retorna True se inicio e fim estao d-separados (sem caminho ativo).

    Parametros
    ----------
    modelo     : BayesianNetwork do pgmpy
    inicio     : nome do no de partida
    fim        : nome do no de destino
    observados : lista de nos observados (evidencias + nos fixados)

    Retorna
    -------
    bool : True = d-separados (sem caminho ativo), False = d-conectados
    """
    ativos = modelo.active_trail_nodes(inicio, observed=observados)[inicio]
    return fim not in ativos

# comparando métodos

def comparar_metodos(
    infer,
    alvos: list,
    evidencia: dict,
    estados: dict,
    priors: dict,
    k: int = 3,
) -> dict:
    """
    Roda K-MRE, K-MAP e K-SIMP e retorna os resultados lado a lado.

    Reproduce a estrutura das Tabelas 3, 5, 6 e 7 do paper.

    Parametros
    ----------
    infer, alvos, evidencia, estados, priors : ver todas_instanciacoes()
    k : numero de explicacoes por metodo

    Retorna
    -------
    dict com chaves 'kmre', 'kmap', 'ksimp', cada uma sendo uma lista de dicts
    """
    return {
        'kmre':  kmre(infer, alvos, evidencia, estados, priors, k=k),
        'kmap':  kmap(infer, alvos, evidencia, estados, k=k),
        'ksimp': ksimp(infer, alvos, evidencia, estados, priors, k=k),
    }


def imprimir_comparativo(resultado: dict, titulo: str = '') -> None:
    """
    Imprime o comparativo de metodos no formato das tabelas do paper.

    Parametros
    ----------
    resultado : saida de comparar_metodos()
    titulo    : cabecalho opcional
    """
    if titulo:
        print(f'\n{"=" * 60}')
        print(f'  {titulo}')
        print(f'{"=" * 60}')

    print(f'\n{"K-MRE":^10} (score = GBF)')
    for r in resultado['kmre']:
        print(f'  GBF={r["gbf"]:8.4f}  [{jeffreys(r["gbf"]):<20}]  {r["label"]}')

    print(f'\n{"K-MAP":^10} (score = probabilidade posterior)')
    for r in resultado['kmap']:
        print(f'  P={r["p_post"]:9.4f}                          {r["label"]}')

    print(f'\n{"K-SIMP":^10} (score = likelihood)')
    for r in resultado['ksimp']:
        print(f'  L={r["likelihood"]:9.4f}                          {r["label"]}')
    print()


# redes de exemplo

def get_example_model(nome: str, n_trilhas: int = 1):
    """
    Retorna uma rede de exemplo pronta para uso com a biblioteca.

    Equivalente ao get_example_model() do pgmpy, mas para as redes
    do paper Yuan, Lim & Lu (2011).

    Redes disponíveis
    -----------------
    'circuito'  — Rede de diagnóstico de circuito elétrico (Poole & Provan, 1991)
                  Seções 3.2.1 e 5.1 do paper
    'viagem'    — Rede de férias do Sr. Smith (Shimony, 1993)
                  Seções 3.2.2 e 5.2 do paper
                  Use n_trilhas=1 (default) para 1 estado de escalada,
                  ou n_trilhas=N (ex: 100) para o modelo multi-estado
    'academe'   — Rede de desempenho acadêmico (Flores et al., 2005)
                  Seções 5.3 do paper

    Parametros
    ----------
    nome      : str — nome da rede ('circuito', 'viagem' ou 'academe')
    n_trilhas : int — apenas para 'viagem'; numero de trilhas de escalada

    Retorna
    -------
    dict com:
        'modelo'    : BayesianNetwork do pgmpy
        'infer'     : VariableElimination pronto para uso
        'alvos'     : lista de variaveis-alvo (nós de diagnóstico)
        'estados'   : dict {variavel: [lista de estados]}
        'priors'    : dict {variavel: {estado: probabilidade}}
        'evidencias': dict com exemplos de evidencias do paper
        'descricao' : str descrevendo a rede
    """
    nome = nome.lower().strip()
    if nome == 'circuito':
        return _exemplo_circuito()
    elif nome == 'viagem':
        return _exemplo_viagem(n_trilhas)
    elif nome == 'academe':
        return _exemplo_academe()
    else:
        disponiveis = ['circuito', 'viagem', 'academe']
        raise ValueError(
            f"Rede '{nome}' nao encontrada. "
            f"Disponiveis: {disponiveis}"
        )


def listar_exemplos() -> None:
    """Lista as redes de exemplo disponíveis com suas descrições."""
    print("Redes de exemplo disponíveis em mre_lib:")
    print()
    print("  'circuito'  — Circuito elétrico (Poole & Provan, 1991)")
    print("                4 gates (A,B,C,D), evidência: corrente detectada")
    print("                Reproduz Tabela 2 e 3 do paper")
    print()
    print("  'viagem'    — Férias do Sr. Smith (Shimony, 1993)")
    print("                n_trilhas=1 (default) ou n_trilhas=N (multi-estado)")
    print("                Reproduz Tabelas 4 e 5 do paper")
    print()
    print("  'academe'   — Desempenho acadêmico (Flores et al., 2005)")
    print("                4 alvos: Teoria, Prática, Extra, OutrosFatores")
    print("                Reproduz Tabela 6 do paper")
    print()
    print("Uso: modelo = get_example_model('circuito')")


def _exemplo_circuito():
    """Rede do Circuito Elétrico — Seções 3.2.1 e 5.1."""
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
    except ImportError:
        raise ImportError("pgmpy e necessario: pip install pgmpy")

    S = ['ok', 'defective']
    F = ['noCurr', 'current']

    modelo = DiscreteBayesianNetwork([
        ('Input', 'OutA'), ('A', 'OutA'),
        ('Input', 'OutB'), ('B', 'OutB'),
        ('OutB',  'OutC'), ('C', 'OutC'),
        ('OutB',  'OutD'), ('D', 'OutD'),
        ('OutA',  'TotalOutput'),
        ('OutC',  'TotalOutput'),
        ('OutD',  'TotalOutput'),
    ])

    modelo.add_cpds(
        TabularCPD('A', 2, [[1-0.016], [0.016]], state_names={'A': S}),
        TabularCPD('B', 2, [[1-0.1],   [0.1]],   state_names={'B': S}),
        TabularCPD('C', 2, [[1-0.15],  [0.15]],  state_names={'C': S}),
        TabularCPD('D', 2, [[1-0.1],   [0.1]],   state_names={'D': S}),
        TabularCPD('Input', 2, [[0.5], [0.5]],   state_names={'Input': F}),

        TabularCPD('OutA', 2,
            [[1.0, 1.0, 1.0, 1-0.999], [0.0, 0.0, 0.0, 0.999]],
            evidence=['A', 'Input'], evidence_card=[2, 2],
            state_names={'OutA': F, 'A': S, 'Input': F}),

        TabularCPD('OutB', 2,
            [[1.0, 1.0, 1.0, 1-0.99], [0.0, 0.0, 0.0, 0.99]],
            evidence=['B', 'Input'], evidence_card=[2, 2],
            state_names={'OutB': F, 'B': S, 'Input': F}),

        TabularCPD('OutC', 2,
            [[1.0, 1.0, 1.0, 1-0.985], [0.0, 0.0, 0.0, 0.985]],
            evidence=['C', 'OutB'], evidence_card=[2, 2],
            state_names={'OutC': F, 'C': S, 'OutB': F}),

        TabularCPD('OutD', 2,
            [[1.0, 1.0, 1.0, 1-0.995], [0.0, 0.0, 0.0, 0.995]],
            evidence=['D', 'OutB'], evidence_card=[2, 2],
            state_names={'OutD': F, 'D': S, 'OutB': F}),

        TabularCPD('TotalOutput', 2,
            [[1-v for v in _noisy_or_circuito()], _noisy_or_circuito()],
            evidence=['OutA', 'OutC', 'OutD'], evidence_card=[2, 2, 2],
            state_names={
                'TotalOutput': F, 'OutA': F, 'OutC': F, 'OutD': F
            }),
    )
    assert modelo.check_model()

    alvos  = ['A', 'B', 'C', 'D']
    priors_raw = {'A': 0.016, 'B': 0.1, 'C': 0.15, 'D': 0.1}

    return {
        'modelo': modelo,
        'infer':  VariableElimination(modelo),
        'alvos':  alvos,
        'estados': {g: S for g in alvos},
        'priors':  {g: {'defective': priors_raw[g], 'ok': 1 - priors_raw[g]}
                    for g in alvos},
        'evidencias': {
            'corrente': {'Input': 'current', 'TotalOutput': 'current'},
        },
        'descricao': (
            "Circuito elétrico com 4 gates (A,B,C,D) — Poole & Provan (1991).\n"
            "Evidência padrão: Input=current, TotalOutput=current.\n"
            "K-MRE reproduz Tabela 2 do paper: (notB,notC)=42.62, "
            "(notA)=39.44, (notB,notD)=35.88."
        ),
    }


def _noisy_or_circuito():
    """Calcula os valores do noisy-OR do TotalOutput."""
    q = {'OutA': 0.9, 'OutC': 0.99, 'OutD': 0.995}
    p_curr = []
    for oa in [0, 1]:
        for oc in [0, 1]:
            for od in [0, 1]:
                if not oa and not oc and not od:
                    p_curr.append(0.0)
                else:
                    p = 1.0
                    if oa: p *= (1 - q['OutA'])
                    if oc: p *= (1 - q['OutC'])
                    if od: p *= (1 - q['OutD'])
                    p_curr.append(1 - p)
    return p_curr


def _exemplo_viagem(n_trilhas: int = 1):
    """Rede de Férias — Shimony (1993). n_trilhas=1 ou N para multi-estado."""
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
    except ImportError:
        raise ImportError("pgmpy e necessario: pip install pgmpy")

    N = max(1, int(n_trilhas))
    multi = N > 1

    if not multi:
        loc_states = ['casa', 'trilha']
        loc_vals   = [[0.8, 0.1], [0.2, 0.9]]
        vivo_vals  = [
            [0.10, 0.90, 0.01, 0.01],
            [0.90, 0.10, 0.99, 0.99],
        ]
    else:
        loc_states = ['casa'] + [f't{i+1}' for i in range(N)]
        p_trilha_nao = (1 - 0.8) / N
        p_trilha_sim = (1 - 0.1) / N
        loc_vals = [[0.8, 0.1]] + [[p_trilha_nao, p_trilha_sim]] * N

        morreu_por_local = [0.10] + [0.90] * N
        viveu_por_local  = [0.90] + [0.10] * N
        # Cols: (Saudavel=nao, loc_1), ..., (Saudavel=nao, loc_N+1),
        #       (Saudavel=sim, loc_1), ..., (Saudavel=sim, loc_N+1)
        linha_morreu = morreu_por_local + [0.01] * (N + 1)
        linha_viveu  = viveu_por_local  + [0.99] * (N + 1)
        vivo_vals = [linha_morreu, linha_viveu]

    modelo = DiscreteBayesianNetwork([
        ('Saudavel', 'Local'),
        ('Saudavel', 'Vivo'),
        ('Local',    'Vivo'),
    ])
    modelo.add_cpds(
        TabularCPD('Saudavel', 2, [[0.2], [0.8]],
                   state_names={'Saudavel': ['nao', 'sim']}),
        TabularCPD('Local', len(loc_states), loc_vals,
                   evidence=['Saudavel'], evidence_card=[2],
                   state_names={'Local': loc_states, 'Saudavel': ['nao', 'sim']}),
        TabularCPD('Vivo', 2, vivo_vals,
                   evidence=['Saudavel', 'Local'],
                   evidence_card=[2, len(loc_states)],
                   state_names={
                       'Vivo':    ['nao', 'sim'],
                       'Saudavel': ['nao', 'sim'],
                       'Local':   loc_states,
                   }),
    )
    assert modelo.check_model()

    alvos   = ['Saudavel', 'Local']
    p_nao   = 0.2
    p_sim   = 0.8
    p_casa  = 0.8 * p_nao + 0.1 * p_sim   # P(Local=casa)
    priors_local = {'casa': p_casa}
    if not multi:
        priors_local['trilha'] = 1 - p_casa
    else:
        p_cada_trilha = (1 - p_casa) / N
        for s in loc_states[1:]:
            priors_local[s] = p_cada_trilha

    return {
        'modelo': modelo,
        'infer':  VariableElimination(modelo),
        'alvos':  alvos,
        'estados': {
            'Saudavel': ['nao', 'sim'],
            'Local':    loc_states,
        },
        'priors': {
            'Saudavel': {'nao': 0.2, 'sim': 0.8},
            'Local':    priors_local,
        },
        'evidencias': {
            'vivo':   {'Vivo': 'sim'},
            'morreu': {'Vivo': 'nao'},
        },
        'descricao': (
            f"Ferias do Sr. Smith — Shimony (1993).\n"
            f"Modelo {'multi-estado (' + str(N) + ' trilhas)' if multi else '1-estado'}.\n"
            "Evidencias: 'vivo' ou 'morreu'.\n"
            "K-MRE reproduz Tabela 5 do paper: (nao-saudavel,trilha)=36.00 "
            "(1-estado), (nao-saudavel)=26.00 (multi-estado)."
        ),
    }


def _exemplo_academe():
    """Rede Academe — Flores et al. (2005)."""
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.inference import VariableElimination
    except ImportError:
        raise ImportError("pgmpy e necessario: pip install pgmpy")

    NOTAS3 = ['boa', 'media', 'ruim']

    modelo = DiscreteBayesianNetwork([
        ('Teoria',        'NotaTP'),
        ('Pratica',       'NotaTP'),
        ('NotaTP',        'NotaGlobal'),
        ('Extra',         'NotaGlobal'),
        ('NotaGlobal',    'NotaFinal'),
        ('OutrosFatores', 'NotaFinal'),
    ])

    modelo.add_cpds(
        TabularCPD('Teoria', 3, [[0.4], [0.3], [0.3]],
                   state_names={'Teoria': NOTAS3}),

        TabularCPD('Pratica', 3, [[0.6], [0.25], [0.15]],
                   state_names={'Pratica': NOTAS3}),

        TabularCPD('Extra', 2, [[0.3], [0.7]],
                   state_names={'Extra': ['sim', 'nao']}),

        TabularCPD('OutrosFatores', 2, [[0.8], [0.2]],
                   state_names={'OutrosFatores': ['mais', 'menos']}),

        TabularCPD('NotaTP', 2,
            [[1.0, 0.85, 0.0, 0.90, 0.20, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.15, 1.0, 0.10, 0.80, 1.0, 1.0, 1.0, 1.0]],
            evidence=['Teoria', 'Pratica'], evidence_card=[3, 3],
            state_names={'NotaTP': ['passa', 'reprova'],
                         'Teoria': NOTAS3, 'Pratica': NOTAS3}),

        TabularCPD('NotaGlobal', 2,
            [[1.0, 1.0, 0.25, 0.0],
             [0.0, 0.0, 0.75, 1.0]],
            evidence=['NotaTP', 'Extra'], evidence_card=[2, 2],
            state_names={'NotaGlobal': ['passa', 'reprova'],
                         'NotaTP': ['passa', 'reprova'],
                         'Extra': ['sim', 'nao']}),

        TabularCPD('NotaFinal', 2,
            [[1.0, 0.70, 0.05, 0.0],
             [0.0, 0.30, 0.95, 1.0]],
            evidence=['NotaGlobal', 'OutrosFatores'], evidence_card=[2, 2],
            state_names={'NotaFinal': ['passa', 'reprova'],
                         'NotaGlobal': ['passa', 'reprova'],
                         'OutrosFatores': ['mais', 'menos']}),
    )
    assert modelo.check_model()

    alvos = ['Teoria', 'Pratica', 'Extra', 'OutrosFatores']

    return {
        'modelo': modelo,
        'infer':  VariableElimination(modelo),
        'alvos':  alvos,
        'estados': {
            'Teoria':        NOTAS3,
            'Pratica':       NOTAS3,
            'Extra':         ['sim', 'nao'],
            'OutrosFatores': ['mais', 'menos'],
        },
        'priors': {
            'Teoria':        {'boa': 0.4, 'media': 0.3, 'ruim': 0.3},
            'Pratica':       {'boa': 0.6, 'media': 0.25, 'ruim': 0.15},
            'Extra':         {'sim': 0.3, 'nao': 0.7},
            'OutrosFatores': {'mais': 0.8, 'menos': 0.2},
        },
        'evidencias': {
            'reprovado': {'NotaFinal': 'reprova'},
            'aprovado':  {'NotaFinal': 'passa'},
        },
        'descricao': (
            "Rede de desempenho academico — Flores et al. (2005).\n"
            "Alvos: Teoria, Pratica, Extra, OutrosFatores.\n"
            "Evidencia padrao: NotaFinal=reprova.\n"
            "K-MRE reproduz Tabela 6 do paper: (ruim teoria)=3.02, "
            "(ruim pratica, nao extra)=2.30."
        ),
    }

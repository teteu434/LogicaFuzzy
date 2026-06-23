"""
fuzzy.py
========
Nucleo do controlador fuzzy de tempo de semaforo.

Este modulo implementa, usando a biblioteca scikit-fuzzy (skfuzzy), as tres
etapas de um sistema de Logica Fuzzy descritas no resumo em PDF:

    1) Fuzzificacao   -> funcoes de pertinencia que convertem valores
                          numericos (QueueSec, WaitTimeSec) em graus [0, 1].
    2) Inferencia      -> regras "SE...ENTAO..." combinadas com a Norma T
                          (E logico fuzzy = minimo) e agregadas com a
                          Norma S (OU logico fuzzy = maximo).
    3) Defuzzificacao  -> o conjunto fuzzy de saida (GreenAdj) e reduzido a
                          um numero preciso pelo metodo do Centroide.

Variaveis do problema (extraidas da tabela do trabalho pratico):

    Entrada  QueueSec     0 a 30 veiculos   {Baixa, Media, Alta}
    Entrada  WaitTimeSec  0 a 120 s         {Curto, Medio, Longo}
    Saida    GreenAdj     -10 a +30 s       {Reduzir, Manter, Aumentar}

O modulo expoe duas formas de calcular a saida, propositalmente:

    * Uma via "manual" (fuzzificacao/inferencia/defuzzificacao explicitas,
      usando apenas as primitivas fuz.trimf/trapmf/interp_membership/defuzz
      do skfuzzy) - usada para gerar os graficos didaticos que mostram cada
      etapa separadamente.
    * Uma via "alta nivel" com skfuzzy.control (Antecedent/Consequent/Rule/
      ControlSystem), que e a forma "profissional" de usar a biblioteca,
      equivalente aos arquivos .fis do Matlab citados no PDF.

As duas vias usam a MESMA base de regras (RULE_DEFINITIONS) e devem produzir
o mesmo resultado numerico - isso e verificado em main.py como uma forma de
validacao cruzada.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Universos de discurso (o "dominio" de cada variavel, conforme a tabela do
# trabalho). Cada universo e um vetor de pontos onde as funcoes de
# pertinencia serao avaliadas.
# ---------------------------------------------------------------------------
UNIVERSE_QUEUE = np.arange(0, 30 + 1, 1)     # QueueSec: 0 a 30 veiculos
UNIVERSE_WAIT = np.arange(0, 120 + 1, 1)     # WaitTimeSec: 0 a 120 segundos
UNIVERSE_GREEN = np.arange(-10, 30 + 1, 1)   # GreenAdj: -10 a +30 segundos

# ---------------------------------------------------------------------------
# Base de regras linguisticas (paralelo com a etapa de "Inferencia Fuzzy" do
# PDF). Cada tupla e uma regra: (termo de QueueSec, termo de WaitTimeSec,
# termo de GreenAdj), equivalente a:
#       SE QueueSec é <termo1> E WaitTimeSec é <termo2> ENTAO GreenAdj é <termo3>
#
# Logica de projeto: fila baixa e espera curta -> sobra tempo de verde na via
# secundaria, entao reduz-se o ajuste; fila alta ou espera longa -> a via
# secundaria esta congestionada, entao aumenta-se o tempo de verde. Os casos
# intermediarios mantêm o tempo de verde atual. Sao 9 regras, cobrindo todas
# as combinacoes possiveis (3 termos x 3 termos), bem acima do minimo de 6
# regras pedido no trabalho.
# ---------------------------------------------------------------------------
RULE_DEFINITIONS = [
    ("Baixa", "Curto", "Reduzir"),
    ("Baixa", "Medio", "Reduzir"),
    ("Baixa", "Longo", "Manter"),
    ("Media", "Curto", "Manter"),
    ("Media", "Medio", "Manter"),
    ("Media", "Longo", "Aumentar"),
    ("Alta", "Curto", "Manter"),
    ("Alta", "Medio", "Aumentar"),
    ("Alta", "Longo", "Aumentar"),
]


def build_variables():
    """Cria as variaveis fuzzy (Antecedent/Consequent) e define suas funcoes
    de pertinencia.

    Esta funcao e a materializacao da etapa de "Fuzzificacao" do PDF: aqui
    decidimos COMO um valor numerico de entrada (ex.: 18 veiculos) vira um
    grau de pertinencia aos conjuntos fuzzy "Baixa", "Media" ou "Alta".

    Foram usadas funcoes trapezoidais (fuz.trapmf) nos termos das pontas
    (ex.: "Baixa", "Curto", "Reduzir", "Aumentar") - que tem um platô onde a
    pertinencia e maxima e constante - e funcoes triangulares (fuz.trimf)
    nos termos centrais (ex.: "Media", "Medio", "Manter") - que sobem e
    descem em torno de um unico pico. Os termos vizinhos se sobrepoem
    propositalmente (transicao gradual), em contraste com a logica booleana,
    que teria cortes abruptos (conforme a comparacao feita no PDF).

    Retorna os tres objetos (queue, wait, green) para serem reutilizados na
    construcao das regras e nos graficos.
    """
    queue = ctrl.Antecedent(UNIVERSE_QUEUE, "QueueSec")
    wait = ctrl.Antecedent(UNIVERSE_WAIT, "WaitTimeSec")
    green = ctrl.Consequent(UNIVERSE_GREEN, "GreenAdj")

    # QueueSec (0-30 veiculos): fila baixa termina por volta de 15, fila alta
    # comeca por volta de 15, com uma faixa "Media" sobreposta entre elas.
    queue["Baixa"] = fuzz.trapmf(queue.universe, [0, 0, 7, 15])
    queue["Media"] = fuzz.trimf(queue.universe, [7, 15, 23])
    queue["Alta"] = fuzz.trapmf(queue.universe, [15, 23, 30, 30])

    # WaitTimeSec (0-120 s): mesma logica de sobreposicao, em outra escala.
    wait["Curto"] = fuzz.trapmf(wait.universe, [0, 0, 30, 60])
    wait["Medio"] = fuzz.trimf(wait.universe, [30, 60, 90])
    wait["Longo"] = fuzz.trapmf(wait.universe, [60, 90, 120, 120])

    # GreenAdj (-10 a +30 s): "Manter" fica centrado em 0 (sem ajuste), com
    # "Reduzir" cobrindo os valores negativos e "Aumentar" cobrindo a maior
    # parte da faixa positiva (a via secundaria normalmente precisa de mais
    # alivio do que de reducao).
    green["Reduzir"] = fuzz.trapmf(green.universe, [-10, -10, -5, 5])
    green["Manter"] = fuzz.trimf(green.universe, [-5, 5, 15])
    green["Aumentar"] = fuzz.trapmf(green.universe, [5, 15, 30, 30])

    return queue, wait, green


def build_rules(queue, wait, green):
    """Traduz RULE_DEFINITIONS em objetos ctrl.Rule do skfuzzy.

    O operador "&" entre os antecedentes implementa a Norma T (T-norma) -
    no skfuzzy, por padrao, o "E" logico fuzzy e o minimo entre os graus de
    pertinencia, exatamente como descrito no PDF (Norma T = min(a, b)).
    """
    rules = []
    for queue_term, wait_term, green_term in RULE_DEFINITIONS:
        rules.append(ctrl.Rule(queue[queue_term] & wait[wait_term], green[green_term]))
    return rules


def build_controller():
    """Monta o sistema fuzzy completo no estilo "alto nivel" do skfuzzy
    (equivalente a um arquivo .fis do Matlab citado no PDF).

    Retorna a simulacao (ja pronta para receber entradas) e as tres
    variaveis fuzzy, para que main.py e visualization.py possam reutiliza-las.
    """
    queue, wait, green = build_variables()
    rules = build_rules(queue, wait, green)
    system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(system)
    return simulation, queue, wait, green


def compute_greenadj_ctrl(simulation, queue_val, wait_val):
    """Calcula GreenAdj usando a API de alto nivel do skfuzzy.control.

    Por baixo dos panos, o skfuzzy executa as mesmas tres etapas
    (fuzzificacao -> inferencia -> defuzzificacao por centroide) que
    implementamos manualmente em `evaluate_rules_manual` / `defuzzify_centroid`.
    """
    simulation.input["QueueSec"] = queue_val
    simulation.input["WaitTimeSec"] = wait_val
    simulation.compute()
    return simulation.output["GreenAdj"]


def membership_degrees(variable, value):
    """Etapa de Fuzzificacao (manual): calcula o grau de pertinencia de um
    valor numerico (`value`) em CADA termo linguistico da variavel fuzzy
    (`variable`), usando interpolacao sobre a funcao de pertinencia
    cadastrada (fuz.interp_membership).

    Exemplo: membership_degrees(queue, 18) pode retornar
        {"Baixa": 0.0, "Media": 0.625, "Alta": 0.375}
    significando que 18 veiculos pertence parcialmente a "Media" e a "Alta".
    """
    return {
        term: float(fuzz.interp_membership(variable.universe, variable[term].mf, value))
        for term in variable.terms
    }


def evaluate_rules_manual(queue, wait, green, queue_val, wait_val):
    """Etapa de Inferencia Fuzzy (manual): avalia as 9 regras de
    RULE_DEFINITIONS para uma entrada especifica (queue_val, wait_val).

    Para cada regra "SE QueueSec é A E WaitTimeSec é B ENTAO GreenAdj é C":
        1. Busca o grau de pertinencia de queue_val ao termo A;
        2. Busca o grau de pertinencia de wait_val ao termo B;
        3. Combina os dois graus com a Norma T minimo (o "E" logico fuzzy),
           obtendo a "forca de disparo" (firing strength) da regra.

    Retorna a lista de regras avaliadas (com sua forca de disparo) e os
    dicionarios de graus de pertinencia usados, para fins de visualizacao.
    """
    degrees_queue = membership_degrees(queue, queue_val)
    degrees_wait = membership_degrees(wait, wait_val)

    evaluated = []
    for queue_term, wait_term, green_term in RULE_DEFINITIONS:
        firing_strength = min(degrees_queue[queue_term], degrees_wait[wait_term])
        evaluated.append(
            {
                "description": f"SE QueueSec é {queue_term} E WaitTimeSec é {wait_term} "
                f"ENTÃO GreenAdj é {green_term}",
                "queue_term": queue_term,
                "wait_term": wait_term,
                "green_term": green_term,
                "strength": firing_strength,
            }
        )
    return evaluated, degrees_queue, degrees_wait


def aggregate_output_manual(green, evaluated_rules):
    """Etapa de Inferencia/Agregacao (manual): combina as conclusoes de
    todas as regras em um unico conjunto fuzzy de saida.

    Para cada regra disparada, a "implicacao" corta o termo de saida
    (green_term) na altura da forca de disparo (np.fmin) - isto e, o
    consequente nao pode ser "mais verdadeiro" do que o antecedente que o
    gerou. Em seguida, a "agregacao" combina os cortes de todas as regras
    com a Norma S maximo (np.fmax) - o "OU" logico fuzzy - formando a
    silhueta final do conjunto fuzzy de saida (antes da defuzzificacao).
    """
    aggregated = np.zeros_like(green.universe, dtype=float)
    for rule in evaluated_rules:
        term_mf = green[rule["green_term"]].mf
        clipped = np.fmin(rule["strength"], term_mf)   # implicacao (T-norma min)
        aggregated = np.fmax(aggregated, clipped)       # agregacao (S-norma max)
    return aggregated


def defuzzify_centroid(green, aggregated):
    """Etapa de Defuzzificacao: reduz o conjunto fuzzy agregado a um unico
    numero, usando o metodo do Centroide (centro de massa da area sob a
    curva), exatamente o metodo citado como "um dos mais comuns" no PDF.
    """
    return float(fuzz.defuzz(green.universe, aggregated, "centroid"))


def compute_greenadj_manual(queue, wait, green, queue_val, wait_val):
    """Encadeia as tres etapas manuais (fuzzificacao -> inferencia/agregacao
    -> defuzzificacao) e retorna o valor final de GreenAdj, junto com todos
    os resultados intermediarios (uteis para os graficos explicativos).
    """
    evaluated_rules, degrees_queue, degrees_wait = evaluate_rules_manual(
        queue, wait, green, queue_val, wait_val
    )
    aggregated = aggregate_output_manual(green, evaluated_rules)
    result = defuzzify_centroid(green, aggregated)
    return {
        "queue_val": queue_val,
        "wait_val": wait_val,
        "degrees_queue": degrees_queue,
        "degrees_wait": degrees_wait,
        "evaluated_rules": evaluated_rules,
        "aggregated": aggregated,
        "result": result,
    }


def compute_surface(queue, wait, green, resolution=31):
    """Calcula GreenAdj para uma grade (QueueSec x WaitTimeSec), usando o
    pipeline manual (mais rapido para muitos pontos do que abrir uma
    simulacao ctrl para cada par de entrada).

    Esta grade e a base para a Tarefa "Gere a superficie 3D com os
    antecedentes nos eixos x e y e o consequente no eixo z" do trabalho.

    Retorna tres matrizes 2D (queue_grid, wait_grid, green_grid) no formato
    esperado por matplotlib (Axes3D.plot_surface).
    """
    queue_vals = np.linspace(UNIVERSE_QUEUE.min(), UNIVERSE_QUEUE.max(), resolution)
    wait_vals = np.linspace(UNIVERSE_WAIT.min(), UNIVERSE_WAIT.max(), resolution)
    queue_grid, wait_grid = np.meshgrid(queue_vals, wait_vals)

    green_grid = np.zeros_like(queue_grid)
    for i in range(queue_grid.shape[0]):
        for j in range(queue_grid.shape[1]):
            output = compute_greenadj_manual(
                queue, wait, green, queue_grid[i, j], wait_grid[i, j]
            )
            green_grid[i, j] = output["result"]

    return queue_grid, wait_grid, green_grid

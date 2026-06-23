"""
main.py
=======
Ponto de entrada do projeto. Aqui as funções de fuzzy.py e visualization.py
são orquestradas para cumprir as quatro tarefas do trabalho prático
"Controle de tempo de semáforo":

    1) Definir e exibir os gráficos das funções de pertinência das
       variáveis de entrada (QueueSec, WaitTimeSec) e de saída (GreenAdj).
    2) Construir e exibir a base de regras fuzzy (mínimo de 6 regras).
    3) Gerar a superfície 3D com os antecedentes nos eixos x/y e o
       consequente no eixo z.
    4) Analisar e interpretar os resultados.

Execute com:
    python main.py

Todas as imagens, o GIF e o relatório de análise são salvos em output/.
"""

import os

import numpy as np

import fuzzy
import visualization

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Exemplo numérico usado para demonstrar passo a passo o pipeline fuzzy
# (fuzzificação -> inferência -> defuzzificação) nos gráficos 02, 03 e 04.
# 18 veículos / 80s de espera caem nas faixas de sobreposição "Média/Alta"
# e "Médio/Longo", de propósito, para que várias regras disparem ao mesmo
# tempo e o exemplo fique mais ilustrativo.
EXAMPLE_QUEUE = 18   # veículos na fila da via secundária
EXAMPLE_WAIT = 80    # segundos de espera média


def print_rule_base():
    """Exibe no console a base de regras fuzzy (Tarefa 2)."""
    print("\n=== Base de regras fuzzy (Etapa 2 — Inferência) ===")
    for i, (queue_term, wait_term, green_term) in enumerate(fuzzy.RULE_DEFINITIONS, start=1):
        print(
            f"  R{i}: SE QueueSec é {queue_term} E WaitTimeSec é {wait_term} "
            f"ENTÃO GreenAdj é {green_term}"
        )
    print(f"Total de regras: {len(fuzzy.RULE_DEFINITIONS)} (mínimo exigido pelo trabalho: 6)\n")


def validate_manual_vs_ctrl(simulation, queue, wait, green, queue_val, wait_val):
    """Roda o mesmo par de entradas pelas duas implementações (manual e
    skfuzzy.control) e compara o resultado, como conferência de que o
    pipeline manual (usado para gerar os gráficos explicativos) está
    consistente com a API de alto nível da biblioteca.
    """
    manual = fuzzy.compute_greenadj_manual(queue, wait, green, queue_val, wait_val)
    via_ctrl = fuzzy.compute_greenadj_ctrl(simulation, queue_val, wait_val)

    print("=== Validação cruzada (pipeline manual vs. skfuzzy.control) ===")
    print(f"  Entrada: QueueSec={queue_val} veículos, WaitTimeSec={wait_val} s")
    print(f"  GreenAdj (manual, centroide explícito)              = {manual['result']:.3f} s")
    print(f"  GreenAdj (skfuzzy.control.ControlSystemSimulation)  = {via_ctrl:.3f} s")
    diff = abs(manual["result"] - via_ctrl)
    status = "OK, pipelines equivalentes" if diff < 0.5 else "ATENÇÃO: divergência relevante"
    print(f"  Diferença absoluta: {diff:.4f} s ({status})\n")
    return manual


def analyze_surface(queue_grid, wait_grid, green_grid):
    """Gera o texto de análise/interpretação pedido na Tarefa 4.

    Importante: os números deste texto não são fixos — eles são extraídos
    estatisticamente do próprio array `green_grid` calculado pelo
    controlador, garantindo que a análise reflita o comportamento real da
    superfície gerada (e mude automaticamente se as funções de pertinência
    ou as regras forem ajustadas).
    """
    z = green_grid
    z_min, z_max = z.min(), z.max()
    idx_min = np.unravel_index(np.argmin(z), z.shape)
    idx_max = np.unravel_index(np.argmax(z), z.shape)

    q_at_min, w_at_min = queue_grid[idx_min], wait_grid[idx_min]
    q_at_max, w_at_max = queue_grid[idx_max], wait_grid[idx_max]

    low_mask = queue_grid <= 10
    mid_mask = (queue_grid > 10) & (queue_grid <= 20)
    high_mask = queue_grid > 20

    text = f"""ANÁLISE E INTERPRETAÇÃO DA SUPERFÍCIE DE CONTROLE
====================================================

Faixa observada de GreenAdj na superfície calculada: {z_min:.2f} s a {z_max:.2f} s
(intervalo total permitido pelo projeto: -10 s a +30 s).

- O menor ajuste (mais próximo de reduzir o verde) ocorre perto de
  QueueSec={q_at_min:.1f} veículos e WaitTimeSec={w_at_min:.1f} s, ou seja,
  exatamente na região de fila baixa e espera curta — condição em que a
  via secundária tem pouco tráfego e não precisa de tempo extra de verde.

- O maior ajuste (mais próximo de aumentar o verde) ocorre perto de
  QueueSec={q_at_max:.1f} veículos e WaitTimeSec={w_at_max:.1f} s, ou seja,
  na região de fila alta e espera longa — condição de congestionamento na
  via secundária, em que o controlador reage aumentando o tempo de verde
  para escoar os veículos acumulados.

- Ajuste médio de GreenAdj por faixa de fila:
    QueueSec <= 10 veículos (fila baixa):  {z[low_mask].mean():+.2f} s em média
    10 < QueueSec <= 20 (fila média):      {z[mid_mask].mean():+.2f} s em média
    QueueSec > 20 veículos (fila alta):    {z[high_mask].mean():+.2f} s em média

- A superfície é suave e cresce de forma gradual (sem saltos abruptos) ao
  longo dos dois eixos de entrada: à medida que QueueSec e/ou WaitTimeSec
  aumentam, GreenAdj também tende a aumentar. Essa suavidade é consequência
  direta da sobreposição das funções de pertinência (Fuzzificação) e da
  forma como as 9 regras cobrem todas as combinações possíveis
  (Inferência) — é exatamente a "transição gradual e mais próxima da
  percepção humana" que o PDF contrasta com a "transição brusca e
  irrealista" de um controlador booleano, que teria apenas patamares fixos
  em vez dessa rampa contínua.

- Em comparação com um semáforo de tempo fixo (citado na introdução do
  trabalho como ineficiente em horários de pico ou de baixo movimento),
  este controlador fuzzy ajusta o tempo de verde de forma proporcional ao
  congestionamento real observado: reage de forma mais agressiva quando
  tanto a fila quanto o tempo de espera estão elevados ao mesmo tempo
  (regra R9), e de forma conservadora quando os dois indicadores estão
  baixos (regra R1), passando por ajustes intermediários nos demais casos.
"""
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Construindo variáveis fuzzy, funções de pertinência e base de regras...")
    simulation, queue, wait, green = fuzzy.build_controller()
    print_rule_base()

    print("Gerando gráficos das funções de pertinência (Tarefa 1)...")
    path1 = visualization.plot_membership_functions(queue, wait, green, OUTPUT_DIR)
    print(f"  -> salvo em {path1}")

    print("\nDemonstrando o pipeline fuzzy passo a passo para um exemplo concreto...")
    manual_result = validate_manual_vs_ctrl(
        simulation, queue, wait, green, EXAMPLE_QUEUE, EXAMPLE_WAIT
    )

    path2 = visualization.plot_fuzzification_example(queue, wait, manual_result, OUTPUT_DIR)
    path3 = visualization.plot_inference_rules(manual_result, OUTPUT_DIR)
    path4 = visualization.plot_defuzzification(green, manual_result, OUTPUT_DIR)
    print(f"  -> salvo em {path2}")
    print(f"  -> salvo em {path3}")
    print(f"  -> salvo em {path4}")

    print("\nCalculando a superfície de controle 3D (Tarefa 3, pode levar alguns segundos)...")
    queue_grid, wait_grid, green_grid = fuzzy.compute_surface(queue, wait, green, resolution=31)
    path5 = visualization.plot_surface_3d(queue_grid, wait_grid, green_grid, OUTPUT_DIR)
    print(f"  -> salvo em {path5}")

    print("Gerando GIF animado (rotação 360°) da superfície 3D...")
    path6 = visualization.animate_surface_3d(queue_grid, wait_grid, green_grid, OUTPUT_DIR)
    print(f"  -> salvo em {path6}")

    print("\nGerando análise e interpretação dos resultados (Tarefa 4)...")
    analysis_text = analyze_surface(queue_grid, wait_grid, green_grid)
    print(analysis_text)
    analysis_path = os.path.join(OUTPUT_DIR, "06_analise_resultados.txt")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis_text)
    print(f"  -> salvo em {analysis_path}")

    print("\nConcluído. Todos os arquivos estão na pasta output/.")


if __name__ == "__main__":
    main()

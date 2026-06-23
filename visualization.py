"""
visualization.py
=================
Geracao de todas as imagens (e do GIF) usadas para visualizar o
funcionamento do controlador fuzzy do semaforo. Nenhuma logica fuzzy mora
aqui - este modulo so DESENHA o que o modulo fuzzy.py calculou, mas os
comentarios explicam o que cada grafico representa e por que ele corresponde
a uma das etapas (Fuzzificacao / Inferencia / Defuzzificacao) descritas no
resumo em PDF.

Todas as imagens sao salvas na pasta `output/` (criada automaticamente).
"""

import os

import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica (so salva arquivos)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessario para projecao 3d)


def ensure_output_dir(output_dir):
    """Garante que a pasta de saida exista antes de salvar qualquer imagem."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _plot_variable_mfs(ax, variable, value=None, degrees=None, title=""):
    """Funcao auxiliar: desenha as curvas de pertinencia de UMA variavel
    fuzzy (todos os seus termos linguisticos) em um eixo (`ax`).

    Se `value` for informado, desenha tambem uma linha vertical pontilhada
    nesse ponto e marca (com um "x") o grau de pertinencia de `value` em
    cada termo - isso ilustra visualmente a etapa de Fuzzificacao: como um
    numero exato (ex.: 18 veiculos) se traduz em graus de pertinencia a
    "Baixa", "Media" e "Alta" simultaneamente.
    """
    for term in variable.terms:
        ax.plot(variable.universe, variable[term].mf, linewidth=2, label=term)

    if value is not None:
        ax.axvline(value, color="black", linestyle="--", linewidth=1, alpha=0.7)
        if degrees:
            for term, degree in degrees.items():
                ax.plot(value, degree, "kx", markersize=9, markeredgewidth=2)

    ax.set_title(title)
    ax.set_xlabel(variable.label)
    ax.set_ylabel("Grau de pertinência (μ)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)


def plot_membership_functions(queue, wait, green, output_dir):
    """Tarefa do trabalho: 'Defina as funcoes de pertinencia para as
    variaveis de entrada e saida (exiba os graficos das funcoes de
    pertinencia)'.

    Gera UMA imagem com 3 subgraficos (QueueSec, WaitTimeSec, GreenAdj),
    cada um mostrando os termos linguisticos da variavel (ex.: Baixa/
    Media/Alta) como curvas de pertinencia sobrepostas - o equivalente
    visual da tabela "Variavel / Intervalo / Termos Linguisticos" da
    foto anexada ao trabalho.
    """
    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    _plot_variable_mfs(axes[0], queue, title="Entrada: QueueSec (fila na via secundária)")
    _plot_variable_mfs(axes[1], wait, title="Entrada: WaitTimeSec (tempo médio de espera)")
    _plot_variable_mfs(axes[2], green, title="Saída: GreenAdj (ajuste do tempo de verde)")

    fig.suptitle("Etapa 1 — Fuzzificação: funções de pertinência das variáveis", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "01_funcoes_pertinencia.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_fuzzification_example(queue, wait, manual_result, output_dir):
    """Ilustra a etapa de Fuzzificacao para UM exemplo numerico concreto
    (manual_result vem de fuzzy.compute_greenadj_manual).

    Mostra as mesmas curvas de QueueSec e WaitTimeSec, mas agora com uma
    linha vertical no valor de entrada escolhido e marcadores "x" nos
    pontos onde essa linha cruza cada curva de pertinencia - exatamente o
    que o PDF descreve como "as entradas do sistema sao convertidas em
    graus de pertinencia a conjuntos fuzzy".
    """
    ensure_output_dir(output_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    queue_val = manual_result["queue_val"]
    wait_val = manual_result["wait_val"]

    _plot_variable_mfs(
        axes[0], queue, value=queue_val, degrees=manual_result["degrees_queue"],
        title=f"QueueSec = {queue_val} veículos",
    )
    _plot_variable_mfs(
        axes[1], wait, value=wait_val, degrees=manual_result["degrees_wait"],
        title=f"WaitTimeSec = {wait_val} s",
    )

    fig.suptitle(
        "Etapa 1 — Fuzzificação de um exemplo concreto (entrada numérica → graus μ)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, "02_fuzzificacao_exemplo.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_inference_rules(manual_result, output_dir):
    """Ilustra a etapa de Inferencia Fuzzy: um grafico de barras com a
    "forca de disparo" (firing strength) de CADA uma das 9 regras para o
    exemplo numerico escolhido.

    A forca de disparo de cada regra e o resultado da Norma T minimo entre
    os dois antecedentes (ex.: min(μ_Media(QueueSec), μ_Longo(WaitTimeSec))).
    Barras mais altas indicam regras mais "relevantes" para aquela entrada;
    barras em zero significam que a regra simplesmente nao se aplica a este
    caso. A cor de cada barra indica o termo de saida (Reduzir/Manter/
    Aumentar) que a regra propoe.
    """
    ensure_output_dir(output_dir)
    rules = manual_result["evaluated_rules"]

    color_by_term = {"Reduzir": "tab:red", "Manter": "tab:gray", "Aumentar": "tab:green"}
    labels = [f"R{i+1}\n{r['queue_term']}/{r['wait_term']}" for i, r in enumerate(rules)]
    strengths = [r["strength"] for r in rules]
    colors = [color_by_term[r["green_term"]] for r in rules]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, strengths, color=colors)
    for bar, rule in zip(bars, rules):
        if rule["strength"] > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                rule["green_term"],
                ha="center",
                fontsize=8,
            )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Força de disparo (Norma T = mínimo)")
    ax.set_title(
        f"Etapa 2 — Inferência Fuzzy: disparo das 9 regras\n"
        f"(QueueSec={manual_result['queue_val']}, WaitTimeSec={manual_result['wait_val']})"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "03_inferencia_regras.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_defuzzification(green, manual_result, output_dir):
    """Ilustra a etapa de Defuzzificacao: mostra o conjunto fuzzy de saida
    JA AGREGADO (a silhueta resultante da uniao - Norma S maximo - de todas
    as regras disparadas) e marca o ponto do Centroide, que e o valor
    numerico final de GreenAdj devolvido pelo controlador.

    A area sombreada e a "área sob a função de pertinência da saída fuzzy"
    citada no PDF; o centroide e o "centro de massa" dessa área.
    """
    ensure_output_dir(output_dir)
    aggregated = manual_result["aggregated"]
    result = manual_result["result"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for term in green.terms:
        ax.plot(green.universe, green[term].mf, linestyle="--", alpha=0.4, label=f"{term} (termo)")

    ax.fill_between(green.universe, aggregated, color="tab:blue", alpha=0.4, label="Saída agregada")
    ax.axvline(result, color="black", linewidth=2, label=f"Centroide = {result:.2f} s")

    ax.set_xlabel("GreenAdj (s)")
    ax.set_ylabel("Grau de pertinência (μ)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Etapa 3 — Defuzzificação (método do Centroide)\n"
        f"QueueSec={manual_result['queue_val']}, WaitTimeSec={manual_result['wait_val']} "
        f"→ GreenAdj = {result:.2f} s"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "04_defuzzificacao_centroide.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_surface_3d(queue_grid, wait_grid, green_grid, output_dir):
    """Tarefa do trabalho: 'Gere a superficie 3D com os antecedentes nos
    eixos x e y e o consequente no eixo z'.

    Eixo X = QueueSec, eixo Y = WaitTimeSec, eixo Z = GreenAdj. Cada ponto
    (x, y, z) da superficie e o resultado de todo o pipeline fuzzy
    (fuzzificacao -> inferencia -> defuzzificacao) para aquele par de
    entradas - ou seja, a superficie e o "mapa de decisao" completo do
    controlador, mostrando como o ajuste do tempo de verde varia conforme
    o trafego na via secundaria.
    """
    ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        queue_grid, wait_grid, green_grid, cmap="viridis", edgecolor="none", alpha=0.95
    )
    ax.set_xlabel("QueueSec (veículos)")
    ax.set_ylabel("WaitTimeSec (s)")
    ax.set_zlabel("GreenAdj (s)")
    ax.set_title("Superfície de controle fuzzy: GreenAdj = f(QueueSec, WaitTimeSec)")
    fig.colorbar(surface, ax=ax, shrink=0.6, label="GreenAdj (s)")

    path = os.path.join(output_dir, "05_superficie_3d.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def animate_surface_3d(queue_grid, wait_grid, green_grid, output_dir, n_frames=72, fps=15):
    """Gera um GIF girando a mesma superficie 3D em 360 graus, para facilitar
    a visualizacao do formato completo da superficie de controle (algumas
    regioes de transicao so ficam claras observando a superficie de varios
    angulos).

    Usa matplotlib.animation com o PillowWriter (biblioteca Pillow), sem
    depender de ffmpeg ou de outras ferramentas externas de video.
    """
    ensure_output_dir(output_dir)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        queue_grid, wait_grid, green_grid, cmap="viridis", edgecolor="none", alpha=0.95
    )
    ax.set_xlabel("QueueSec (veículos)")
    ax.set_ylabel("WaitTimeSec (s)")
    ax.set_zlabel("GreenAdj (s)")
    ax.set_title("Superfície de controle fuzzy (rotação 360°)")
    fig.colorbar(surface, ax=ax, shrink=0.6, label="GreenAdj (s)")

    def update(frame):
        ax.view_init(elev=28, azim=frame * (360 / n_frames))
        return (surface,)

    animation = FuncAnimation(fig, update, frames=n_frames, blit=False)
    path = os.path.join(output_dir, "05_superficie_3d.gif")
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path

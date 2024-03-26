import numpy as np
from Ants_simulator.simulator import simulator
from Ants_simulator.config import Config
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter


def main():
    #test()

    result_root_directory = Path("experiments", "experiment_1")
    #experiment_1(result_root_directory)
    plot_heat_maps_exp1(result_root_directory, Config.ANTS)
    plot_line_graph_exp1(result_root_directory, Config.ANTS)

    result_root_directory = Path("experiments", "experiment_2")
    #experiment_2(result_root_directory)
    plot_heat_maps_exp2(result_root_directory, Config.ANTS)
    plot_line_graph_exp2(result_root_directory, Config.ANTS)

    exp3_result_root_directory = Path("experiments", "experiment_3")
    exp2_result_root_directory = Path("experiments", "experiment_2")
    #experiment_3(exp3_result_root_directory)
    plot_heat_maps_exp3(exp3_result_root_directory, Config.ANTS)
    plot_line_graph_exp3(exp3_result_root_directory, Config.ANTS)
    plot_evaluation_exp3(exp3_result_root_directory, exp2_result_root_directory, Config.ANTS)
    plot_improvements_exp3(exp3_result_root_directory, exp2_result_root_directory, Config.ANTS)


def experiment_1(result_root_directory: Path):
    config = Config()
    config.POLICY = 0

    detractors_fracs = [0.1, 0.05, 0.03, 0.01, 0.0]
    evaporation_rates = [0.0, 1.0, 20.0, 100.0, 1000.0]

    for detractors_frac in detractors_fracs:
        for evaporation_rate in evaporation_rates:
            config.K = evaporation_rate
            config.MAL_ANT_FRC = detractors_frac

            result_directory = result_root_directory.joinpath(f"df_{detractors_frac}_er_{evaporation_rate}")
            result_directory.mkdir(exist_ok=True, parents=True)
            collected_foods, delivered_foods, ants_collected_ever, ants_delivered_ever = simulator(config, result_directory)

            np.savetxt(result_directory.joinpath("collected_foods.txt"), collected_foods, fmt='%d')
            np.savetxt(result_directory.joinpath("delivered_foods.txt"), delivered_foods, fmt='%d')
            np.savetxt(result_directory.joinpath("ants_collected_ever.txt"), ants_collected_ever, fmt='%d')
            np.savetxt(result_directory.joinpath("ants_delivered_ever.txt"), ants_delivered_ever, fmt='%d')
            with open(str(result_directory) + "\\config.json", 'w') as file:
                json.dump(config.__dict__, file)


def experiment_2(result_root_directory: Path):
    config = Config()
    config.POLICY = 1
    config.SHOW_CAUT_PHERO = True
    config.SHOW_HOME_PHERO = False
    config.K = 1
    config.MAL_ANT_FRC = 0.03

    patience_refill_steps = [1, 5, 10, 50, 100]
    patience_thresholds = [50, 100, 250, 500, 1000]

    for patience_refill_step in patience_refill_steps:
        for patience_threshold in patience_thresholds:
            config.TP = patience_refill_step
            config.P_MAX = patience_threshold

            result_directory = result_root_directory.joinpath(f"tp_{patience_refill_step}_pm_{patience_threshold}")
            result_directory.mkdir(exist_ok=True, parents=True)
            collected_foods, delivered_foods, ants_collected_ever, ants_delivered_ever = simulator(config,
                                                                                                   result_directory)

            np.savetxt(result_directory.joinpath("collected_foods.txt"), collected_foods, fmt='%d')
            np.savetxt(result_directory.joinpath("delivered_foods.txt"), delivered_foods, fmt='%d')
            np.savetxt(result_directory.joinpath("ants_collected_ever.txt"), ants_collected_ever, fmt='%d')
            np.savetxt(result_directory.joinpath("ants_delivered_ever.txt"), ants_delivered_ever, fmt='%d')
            with open(str(result_directory) + "\\config.json", 'w') as file:
                json.dump(config.__dict__, file)


def experiment_3(result_root_directory: Path):
    config = Config()
    config.POLICY = 2
    config.SHOW_CAUT_PHERO = True
    config.SHOW_HOME_PHERO = False
    config.TP = 5
    config.K = 1
    config.MAL_ANT_FRC = 0.03

    patience_refill_steps = [1, 5, 10, 50, 100]
    patience_thresholds = [50, 100, 250, 500, 1000]

    for patience_refill_step in patience_refill_steps:
        for patience_threshold in patience_thresholds:
            config.TP = patience_refill_step
            config.P_MAX = patience_threshold

            result_directory = result_root_directory.joinpath(f"pr_{patience_refill_step}_pt_{patience_threshold}")
            result_directory.mkdir(exist_ok=True, parents=True)
            collected_foods, delivered_foods, ants_collected_ever, ants_delivered_ever = simulator(config,
                                                                                                   result_directory)

            np.savetxt(result_directory.joinpath("collected_foods.txt"), collected_foods, fmt='%d')
            np.savetxt(result_directory.joinpath("delivered_foods.txt"), delivered_foods, fmt='%d')
            np.savetxt(result_directory.joinpath("ants_collected_ever.txt"), ants_collected_ever, fmt='%d')
            np.savetxt(result_directory.joinpath("ants_delivered_ever.txt"), ants_delivered_ever, fmt='%d')
            with open(str(result_directory) + "\\config.json", 'w') as file:
                json.dump(config.__dict__, file)


def test():
    config = Config()
    config.POLICY = 2
    simulator(config)


def plot_line_graph_exp1(experiment_root_dir, ants_number):
    experiments_dir = experiment_root_dir.iterdir()

    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    er_values = []
    df_values = []
    for experiment_dir in experiments_dir:
        if experiment_dir.is_dir():
            ants_collected = np.loadtxt(experiment_dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(experiment_dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(experiment_dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(experiment_dir.joinpath("delivered_foods.txt"))
            with open(str(experiment_dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                er = config["K"]
                df = config["MAL_ANT_FRC"]
                if er not in er_values:
                    er_values.append(er)
                if df not in df_values:
                    df_values.append(df)
            cooperator_ants_num = int(ants_number * (1 - df))
            avg_collected_foods_dict[(df, er)] = collected_foods / cooperator_ants_num
            avg_delivered_foods_dict[(df, er)] = delivered_foods / cooperator_ants_num
            ants_delivered_frac_dict[(df, er)] = ants_delivered / cooperator_ants_num
            ants_collected_frac_dict[(df, er)] = ants_collected / cooperator_ants_num

    configurations = [(0.0, 0.0), (0.01, 100.0), (0.03, 1.0), (0.05, 100.0)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot the confusion matrices
    x = (np.arange(0, 10000) / 10000) * 100
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        y1 = avg_collected_foods_dict[configurations[i]]
        y2 = avg_delivered_foods_dict[configurations[i]]
        y3 = ants_collected_frac_dict[configurations[i]]
        y4 = ants_delivered_frac_dict[configurations[i]]
        plt.ylim(0.001, 100)
        plt.text(0, 10, highlight_chars[i], fontsize=12, fontweight='bold', color='black')
        l1 = plt.plot(x, y1, color='b', label="foods collected per ant")
        l2 = plt.plot(x, y2, '--', color='orange', label="foods delivered per ant")
        plt.xlabel("percentage of simulation step")
        plt.ylabel("log[average food bits per ant]")
        plt.yscale('log')
        plt.twinx()
        l3 = plt.plot(x, y3, ':', color='g', label="successful collectors")
        l4 = plt.plot(x, y4, '-.', color='r', label="successful delivers")
        plt.ylim(0.001, 100)
        plt.ylabel("log[percentage of cooperators]")
        plt.yscale('log')

    labels = ["foods collected per ant", "foods delivered per ant", "successful collectors", "successful delivers"]
    lgd = fig.legend([l1, l2, l3, l4], labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.tight_layout()
    image = experiment_root_dir.joinpath("line_graph.png")
    plt.savefig(str(image), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_line_graph_exp2(experiment_root_dir, ants_number):
    experiments_dir = experiment_root_dir.iterdir()

    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    tp_values = []
    pm_values = []
    for experiment_dir in experiments_dir:
        if experiment_dir.is_dir():
            ants_collected = np.loadtxt(experiment_dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(experiment_dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(experiment_dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(experiment_dir.joinpath("delivered_foods.txt"))
            with open(str(experiment_dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                tp = config["TP"]
                pm = config["P_MAX"]
                if tp not in tp_values:
                    tp_values.append(tp)
                if pm not in pm_values:
                    pm_values.append(pm)
            cooperator_ants_num = int(ants_number * (1 - config["MAL_ANT_FRC"]))
            avg_collected_foods_dict[(tp, pm)] = collected_foods / cooperator_ants_num
            avg_delivered_foods_dict[(tp, pm)] = delivered_foods / cooperator_ants_num
            ants_delivered_frac_dict[(tp, pm)] = ants_delivered / cooperator_ants_num
            ants_collected_frac_dict[(tp, pm)] = ants_collected / cooperator_ants_num

    configurations = [(1, 250), (5, 1000), (10, 100), (50, 500)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot the confusion matrices
    x = (np.arange(0, 10000) / 10000) * 100
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        y1 = avg_collected_foods_dict[configurations[i]]
        y2 = avg_delivered_foods_dict[configurations[i]]
        y3 = ants_collected_frac_dict[configurations[i]]
        y4 = ants_delivered_frac_dict[configurations[i]]
        plt.ylim(0.001, 100)
        plt.text(0, 10, highlight_chars[i], fontsize=12, fontweight='bold', color='black')
        l1 = plt.plot(x, y1, color='b', label="foods collected per ant")
        l2 = plt.plot(x, y2, '--', color='orange', label="foods delivered per ant")
        plt.xlabel("percentage of simulation step")
        plt.ylabel("log[average food bits per ant]")
        plt.yscale('log')
        plt.twinx()
        l3 = plt.plot(x, y3, ':', color='g', label="successful collectors")
        l4 = plt.plot(x, y4, '-.', color='r', label="successful delivers")
        plt.ylim(0.001, 100)
        plt.ylabel("log[percentage of cooperators]")
        plt.yscale('log')

    labels = ["foods collected per ant", "foods delivered per ant", "successful collectors", "successful delivers"]
    lgd = fig.legend([l1, l2, l3, l4], labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.tight_layout()
    image = experiment_root_dir.joinpath("line_graph.png")
    plt.savefig(str(image), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_line_graph_exp3(experiment_root_dir, ants_number):
    experiments_dir = experiment_root_dir.iterdir()

    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    tp_values = []
    pm_values = []
    for experiment_dir in experiments_dir:
        if experiment_dir.is_dir():
            ants_collected = np.loadtxt(experiment_dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(experiment_dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(experiment_dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(experiment_dir.joinpath("delivered_foods.txt"))
            with open(str(experiment_dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                tp = config["TP"]
                pm = config["P_MAX"]
                if tp not in tp_values:
                    tp_values.append(tp)
                if pm not in pm_values:
                    pm_values.append(pm)
            cooperator_ants_num = int(ants_number * (1 - config["MAL_ANT_FRC"]))
            avg_collected_foods_dict[(tp, pm)] = collected_foods / cooperator_ants_num
            avg_delivered_foods_dict[(tp, pm)] = delivered_foods / cooperator_ants_num
            ants_delivered_frac_dict[(tp, pm)] = ants_delivered / cooperator_ants_num
            ants_collected_frac_dict[(tp, pm)] = ants_collected / cooperator_ants_num

    configurations = [(1, 250), (5, 1000), (10, 100), (50, 500)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot the confusion matrices
    x = (np.arange(0, 10000) / 10000) * 100
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        y1 = avg_collected_foods_dict[configurations[i]]
        y2 = avg_delivered_foods_dict[configurations[i]]
        y3 = ants_collected_frac_dict[configurations[i]]
        y4 = ants_delivered_frac_dict[configurations[i]]
        plt.ylim(0.001, 100)
        plt.text(0, 10, highlight_chars[i], fontsize=12, fontweight='bold', color='black')
        l1 = plt.plot(x, y1, color='b', label="foods collected per ant")
        l2 = plt.plot(x, y2, '--', color='orange', label="foods delivered per ant")
        plt.xlabel("percentage of simulation step")
        plt.ylabel("log[average food bits per ant]")
        plt.yscale('log')
        plt.twinx()
        l3 = plt.plot(x, y3, ':', color='g', label="successful collectors")
        l4 = plt.plot(x, y4, '-.', color='r', label="successful delivers")
        plt.ylim(0.001, 100)
        plt.ylabel("log[percentage of cooperators]")
        plt.yscale('log')

    labels = ["foods collected per ant", "foods delivered per ant", "successful collectors", "successful delivers"]
    lgd = fig.legend([l1, l2, l3, l4], labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.tight_layout()
    image = experiment_root_dir.joinpath("line_graph.png")
    plt.savefig(str(image), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_heat_maps_exp1(experiment_root_dir: Path, ants_number):
    experiments_dir = experiment_root_dir.iterdir()

    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    er_values = []
    df_values = []
    for experiment_dir in experiments_dir:
        if experiment_dir.is_dir():
            ants_collected = np.loadtxt(experiment_dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(experiment_dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(experiment_dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(experiment_dir.joinpath("delivered_foods.txt"))
            with open(str(experiment_dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                er = config["K"]
                df = config["MAL_ANT_FRC"]
                if er not in er_values:
                    er_values.append(er)
                if df not in df_values:
                    df_values.append(df)
            cooperator_ants_num = int(ants_number * (1 - df))
            avg_collected_foods_dict[(df, er)] = collected_foods[-1] / cooperator_ants_num
            avg_delivered_foods_dict[(df, er)] = delivered_foods[-1] / cooperator_ants_num
            ants_delivered_frac_dict[(df, er)] = ants_delivered[-1] / cooperator_ants_num
            ants_collected_frac_dict[(df, er)] = ants_collected[-1] / cooperator_ants_num

    # Convert the dictionary of results to a confusion matrix
    collected_foods_matrix = np.zeros((len(df_values), len(er_values)))
    delivered_foods_matrix = np.zeros((len(df_values), len(er_values)))
    ants_collected_ever_matrix = np.zeros((len(df_values), len(er_values)))
    ants_delivered_ever_matrix = np.zeros((len(df_values), len(er_values)))
    df_values.sort()
    er_values.sort()
    for j, df in enumerate(df_values):
        for i, er in enumerate(er_values):
            collected_foods_matrix[i, j] = avg_collected_foods_dict[(df, er)]
            delivered_foods_matrix[i, j] = avg_delivered_foods_dict[(df, er)]
            ants_collected_ever_matrix[i, j] = ants_collected_frac_dict[(df, er)]
            ants_delivered_ever_matrix[i, j] = ants_delivered_frac_dict[(df, er)]
    # Plot the confusion matrices
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.xlabel("percentage of detractors in the colony", labelpad=20)
    plt.ylabel("evaporation rate multiplier", labelpad=50)
    # Define the indices of the pixels you want to highlight
    highlighted_pixels = [(0, 0), (3, 1), (1, 2), (3, 3)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot confusion matrix 1
    plt.subplot(2, 2, 1)
    plt.imshow(collected_foods_matrix, cmap='coolwarm_r')
    plt.xticks([], [])
    plt.yticks(np.arange(len(er_values)), er_values)
    plt.title('(a)')
    plt.clim(0, 30)
    plt.colorbar()

    # Add rectangles and text for highlighted pixels
    for char, (i, j) in zip(highlight_chars, highlighted_pixels):
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j, i, char, ha='center', va='center', color='white', fontsize=12)

    # Plot confusion matrix 2
    plt.subplot(2, 2, 2)
    plt.imshow(delivered_foods_matrix, cmap='coolwarm_r')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('(b)')
    plt.clim(0, 30)
    plt.colorbar()

    # Plot confusion matrix 3
    plt.subplot(2, 2, 3)
    plt.imshow(ants_collected_ever_matrix, cmap='coolwarm_r')
    plt.xticks(np.arange(len(df_values)), df_values)
    plt.yticks(np.arange(len(er_values)), er_values)
    plt.title('(c)')
    plt.clim(0, 1)
    plt.colorbar()
    for char, (i, j) in zip(highlight_chars, highlighted_pixels):
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j, i, char, ha='center', va='center', color='white', fontsize=12)

    # Plot confusion matrix 4
    plt.subplot(2, 2, 4)
    plt.imshow(ants_delivered_ever_matrix, cmap='coolwarm_r')
    plt.xticks(np.arange(len(df_values)), df_values)
    plt.yticks([], [])
    plt.title("(d)")
    plt.clim(0, 1)
    plt.colorbar()

    plt.tight_layout()
    image = experiment_root_dir.joinpath("confusion_matrix.png")
    plt.savefig(str(image))
    plt.show()


def plot_heat_maps_exp2(experiment_root_dir: Path, ants_number):
    experiments_dir = experiment_root_dir.iterdir()

    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    tp_values = []
    pm_values = []
    for experiment_dir in experiments_dir:
        if experiment_dir.is_dir():
            ants_collected = np.loadtxt(experiment_dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(experiment_dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(experiment_dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(experiment_dir.joinpath("delivered_foods.txt"))
            with open(str(experiment_dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                tp = config["TP"]
                pm = config["P_MAX"]
                if tp not in tp_values:
                    tp_values.append(tp)
                if pm not in pm_values:
                    pm_values.append(pm)
            cooperator_ants_num = int(ants_number * (1 - config["MAL_ANT_FRC"]))
            avg_collected_foods_dict[(tp, pm)] = collected_foods[-1] / cooperator_ants_num
            avg_delivered_foods_dict[(tp, pm)] = delivered_foods[-1] / cooperator_ants_num
            ants_delivered_frac_dict[(tp, pm)] = ants_delivered[-1] / cooperator_ants_num
            ants_collected_frac_dict[(tp, pm)] = ants_collected[-1] / cooperator_ants_num

    # Convert the dictionary of results to a confusion matrix
    collected_foods_matrix = np.zeros((len(pm_values), len(tp_values)))
    delivered_foods_matrix = np.zeros((len(pm_values), len(tp_values)))
    ants_collected_ever_matrix = np.zeros((len(pm_values), len(tp_values)))
    ants_delivered_ever_matrix = np.zeros((len(pm_values), len(tp_values)))
    pm_values.sort()
    tp_values.sort()
    for i, pm in enumerate(pm_values):
        for j, tp in enumerate(tp_values):
            collected_foods_matrix[i, j] = avg_collected_foods_dict[(tp, pm)]
            delivered_foods_matrix[i, j] = avg_delivered_foods_dict[(tp, pm)]
            ants_collected_ever_matrix[i, j] = ants_collected_frac_dict[(tp, pm)]
            ants_delivered_ever_matrix[i, j] = ants_delivered_frac_dict[(tp, pm)]
    # Plot the confusion matrices
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.xlabel("patience refill steps", labelpad=20)
    plt.ylabel("maximum patience thresholds", labelpad=50)
    # Define the indices of the pixels you want to highlight
    highlighted_pixels = [(2, 0), (4, 1), (1, 2), (3, 3)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot confusion matrix 1
    plt.subplot(2, 2, 1)
    plt.imshow(collected_foods_matrix, cmap='coolwarm_r')
    plt.xticks([], [])
    plt.yticks(np.arange(len(pm_values)), pm_values)
    plt.title('(a)')
    plt.clim(3, 5)
    plt.colorbar()
    # Add rectangles and text for highlighted pixels
    for char, (i, j) in zip(highlight_chars, highlighted_pixels):
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j, i, char, ha='center', va='center', color='white', fontsize=12)

    # Plot confusion matrix 2
    plt.subplot(2, 2, 2)
    plt.imshow(delivered_foods_matrix, cmap='coolwarm_r')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('(b)')
    plt.clim(3, 5)
    plt.colorbar()

    # Plot confusion matrix 3
    plt.subplot(2, 2, 3)
    plt.imshow(ants_collected_ever_matrix, cmap='coolwarm_r')
    plt.xticks(np.arange(len(tp_values)), tp_values)
    plt.yticks(np.arange(len(pm_values)), pm_values)
    plt.title('(c)')
    plt.clim(0, 1)
    plt.colorbar()
    for char, (i, j) in zip(highlight_chars, highlighted_pixels):
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j, i, char, ha='center', va='center', color='white', fontsize=12)

    # Plot confusion matrix 4
    plt.subplot(2, 2, 4)
    plt.imshow(ants_delivered_ever_matrix, cmap='coolwarm_r')
    plt.xticks(np.arange(len(tp_values)), tp_values)
    plt.yticks([], [])
    plt.title("(d)")
    plt.clim(0, 1)
    plt.colorbar()

    plt.tight_layout()
    image = experiment_root_dir.joinpath("confusion_matrix.png")
    plt.savefig(str(image))
    plt.show()


def plot_heat_maps_exp3(experiment_root_dir: Path, ants_number):
    experiments_dir = experiment_root_dir.iterdir()

    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    tp_values = []
    pm_values = []
    for experiment_dir in experiments_dir:
        if experiment_dir.is_dir():
            ants_collected = np.loadtxt(experiment_dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(experiment_dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(experiment_dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(experiment_dir.joinpath("delivered_foods.txt"))
            with open(str(experiment_dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                tp = config["TP"]
                pm = config["P_MAX"]
                if tp not in tp_values:
                    tp_values.append(tp)
                if pm not in pm_values:
                    pm_values.append(pm)
            cooperator_ants_num = int(ants_number * (1 - config["MAL_ANT_FRC"]))
            avg_collected_foods_dict[(tp, pm)] = collected_foods[-1] / cooperator_ants_num
            avg_delivered_foods_dict[(tp, pm)] = delivered_foods[-1] / cooperator_ants_num
            ants_delivered_frac_dict[(tp, pm)] = ants_delivered[-1] / cooperator_ants_num
            ants_collected_frac_dict[(tp, pm)] = ants_collected[-1] / cooperator_ants_num

    # Convert the dictionary of results to a confusion matrix
    collected_foods_matrix = np.zeros((len(pm_values), len(tp_values)))
    delivered_foods_matrix = np.zeros((len(pm_values), len(tp_values)))
    ants_collected_ever_matrix = np.zeros((len(pm_values), len(tp_values)))
    ants_delivered_ever_matrix = np.zeros((len(pm_values), len(tp_values)))
    pm_values.sort()
    tp_values.sort()
    for i, pm in enumerate(pm_values):
        for j, tp in enumerate(tp_values):
            collected_foods_matrix[i, j] = avg_collected_foods_dict[(tp, pm)]
            delivered_foods_matrix[i, j] = avg_delivered_foods_dict[(tp, pm)]
            ants_collected_ever_matrix[i, j] = ants_collected_frac_dict[(tp, pm)]
            ants_delivered_ever_matrix[i, j] = ants_delivered_frac_dict[(tp, pm)]
    # Plot the confusion matrices
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.xlabel("patience refill steps", labelpad=20)
    plt.ylabel("maximum patience thresholds", labelpad=50)
    # Define the indices of the pixels you want to highlight
    highlighted_pixels = [(2, 0), (4, 1), (1, 2), (3, 3)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot confusion matrix 1
    plt.subplot(2, 2, 1)
    plt.imshow(collected_foods_matrix, cmap='coolwarm_r')
    plt.xticks([], [])
    plt.yticks(np.arange(len(pm_values)), pm_values)
    plt.title('(a)')
    #plt.clim(3, 5)
    plt.colorbar()
    # Add rectangles and text for highlighted pixels
    for char, (i, j) in zip(highlight_chars, highlighted_pixels):
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j, i, char, ha='center', va='center', color='white', fontsize=12)

    # Plot confusion matrix 2
    plt.subplot(2, 2, 2)
    plt.imshow(delivered_foods_matrix, cmap='coolwarm_r')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title('(b)')
    #plt.clim(3, 5)
    plt.colorbar()

    # Plot confusion matrix 3
    plt.subplot(2, 2, 3)
    plt.imshow(ants_collected_ever_matrix, cmap='coolwarm_r')
    plt.xticks(np.arange(len(tp_values)), tp_values)
    plt.yticks(np.arange(len(pm_values)), pm_values)
    plt.title('(c)')
    plt.clim(0, 1)
    plt.colorbar()
    for char, (i, j) in zip(highlight_chars, highlighted_pixels):
        rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(j, i, char, ha='center', va='center', color='white', fontsize=12)

    # Plot confusion matrix 4
    plt.subplot(2, 2, 4)
    plt.imshow(ants_delivered_ever_matrix, cmap='coolwarm_r')
    plt.xticks(np.arange(len(tp_values)), tp_values)
    plt.yticks([], [])
    plt.title("(d)")
    plt.clim(0, 1)
    plt.colorbar()

    plt.tight_layout()
    image = experiment_root_dir.joinpath("confusion_matrix.png")
    plt.savefig(str(image))
    plt.show()


def plot_evaluation_exp3(experiment3_root_dir: Path, experiment2_root_dir: Path, ants_number):
    # load experiment 3 results

    exp3_avg_collected_foods_dict, exp3_avg_delivered_foods_dict,\
    exp3_ants_collected_frac_dict, exp3_ants_delivered_frac_dict = load_results(experiment3_root_dir, ants_number)

    # load experiment 2 results
    exp2_avg_collected_foods_dict, exp2_avg_delivered_foods_dict, \
    exp2_ants_collected_frac_dict, exp2_ants_delivered_frac_dict = load_results(experiment2_root_dir, ants_number)

    configurations = [(1, 250), (5, 1000), (10, 100), (50, 500)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot the confusion matrices
    x = (np.arange(0, 10000) / 10000) * 100
    fig = plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        y1 = exp2_avg_collected_foods_dict[configurations[i]]
        y2 = exp2_avg_delivered_foods_dict[configurations[i]]
        y3 = exp3_avg_collected_foods_dict[configurations[i]]
        y4 = exp3_avg_delivered_foods_dict[configurations[i]]
        plt.ylim(0.01, 100)
        plt.text(0, 10, highlight_chars[i], fontsize=12, fontweight='bold', color='black')
        l1 = plt.plot(x, y1, color='b', label="foods collected per ant (base line)")
        l2 = plt.plot(x, y2, '--', color='orange', label="foods delivered per ant (base line)")
        l3 = plt.plot(x, y3, ':', color='g', label="foods collected per ant (improved)")
        l4 = plt.plot(x, y4, '-.', color='r', label="foods delivered per ant (improved)")
        plt.minorticks_off()
        plt.xlabel("percentage of simulation step")
        plt.ylabel("log[average food bits per ant]")
        plt.yscale('log')

    labels = ["foods collected per ant (base line)", "foods delivered per ant (base line)",
              "foods collected per ant (improved)", "foods delivered per ant (improved)"]
    lgd = fig.legend([l1, l2, l3, l4], labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.tight_layout()
    image = experiment3_root_dir.joinpath("evaluation_graph.png")
    plt.savefig(str(image), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_improvements_exp3(experiment3_root_dir: Path, experiment2_root_dir: Path, ants_number):
    # load experiment 3 results

    exp3_avg_collected_foods_dict, exp3_avg_delivered_foods_dict, \
    exp3_ants_collected_frac_dict, exp3_ants_delivered_frac_dict = load_results(experiment3_root_dir, ants_number)

    # load experiment 2 results
    exp2_avg_collected_foods_dict, exp2_avg_delivered_foods_dict, \
    exp2_ants_collected_frac_dict, exp2_ants_delivered_frac_dict = load_results(experiment2_root_dir, ants_number)

    configurations = [(1, 250), (5, 1000), (10, 100), (50, 500)]
    highlight_chars = ['α', 'β', 'γ', 'σ']
    collected_foods = []
    delivered_foods = []
    improvements = []
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        y1 = exp2_avg_collected_foods_dict[configurations[i]][-1]
        y2 = exp2_avg_delivered_foods_dict[configurations[i]][-1]
        y3 = exp3_avg_collected_foods_dict[configurations[i]][-1]
        y4 = exp3_avg_delivered_foods_dict[configurations[i]][-1]
        collected_foods.append([y1, y3])
        delivered_foods.append([y2, y4])
        improvements.append([100 * ((y3 / y1) - 1), 100 * ((y4 / y2) - 1)])

    x = np.arange(len(highlight_chars))
    # Width of the bars
    bar_width = 0.1
    # Plot the confusion matrices
    fig = plt.figure(figsize=(8, 8))
    # Create the bars
    b1 = plt.bar(x - 2 * bar_width, np.array(collected_foods)[:, 0], bar_width, color='r')
    b2 = plt.bar(x - bar_width, np.array(collected_foods)[:, 1], bar_width, color='g')
    b3 = plt.bar(x + bar_width, np.array(delivered_foods)[:, 0], bar_width, color='b')
    b4 = plt.bar(x + 2 * bar_width, np.array(delivered_foods)[:, 1], bar_width, color='orange')
    plt.xticks([i for i in range(4)], highlight_chars)
    labels = ["collected food bits (base lise)", "collected food bits (improved)",
              "delivered food bits (base line)", "delivered food bits (improved)"]
    lgd = fig.legend([b1, b2, b3, b4], labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2 )
    image = experiment3_root_dir.joinpath("improvement_bar.png")
    plt.savefig(str(image), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    print(improvements)


def load_results(experiment_root_dir: Path, ants_number):
    avg_collected_foods_dict = {}
    avg_delivered_foods_dict = {}
    ants_collected_frac_dict = {}
    ants_delivered_frac_dict = {}
    tp_values = []
    pm_values = []
    experiment_dir = experiment_root_dir.iterdir()
    for dir in experiment_dir:
        if dir.is_dir():
            ants_collected = np.loadtxt(dir.joinpath("ants_collected_ever.txt"))
            ants_delivered = np.loadtxt(dir.joinpath("ants_delivered_ever.txt"))
            collected_foods = np.loadtxt(dir.joinpath("collected_foods.txt"))
            delivered_foods = np.loadtxt(dir.joinpath("delivered_foods.txt"))
            with open(str(dir.joinpath('config.json')), 'r') as f:
                config = json.load(f)
                tp = config["TP"]
                pm = config["P_MAX"]
                if tp not in tp_values:
                    tp_values.append(tp)
                if pm not in pm_values:
                    pm_values.append(pm)
            cooperator_ants_num = int(ants_number * (1 - config["MAL_ANT_FRC"]))
            avg_collected_foods_dict[(tp, pm)] = collected_foods / cooperator_ants_num
            avg_delivered_foods_dict[(tp, pm)] = delivered_foods / cooperator_ants_num
            ants_delivered_frac_dict[(tp, pm)] = ants_delivered / cooperator_ants_num
            ants_collected_frac_dict[(tp, pm)] = ants_collected / cooperator_ants_num
    return avg_collected_foods_dict, avg_delivered_foods_dict, ants_collected_frac_dict, ants_delivered_frac_dict


if __name__ == "__main__":
    main()



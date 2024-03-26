import numpy as np
from Ants_simulator.simulator import simulator
from Ants_simulator.config import Config
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    do_experiments()
    #test()
    #plot()


def do_experiments():
    """
    This functions runs the experiments
    :return:
    """
    exp1_result_root_directory = Path("results", "experiment_1")
    exp2_result_root_directory = Path("results", "experiment_2")
    exp3_result_root_directory = Path("results", "experiment_3")
    # 20 trial for each experiment
    for t in range(20):
        result_root_directory = exp1_result_root_directory.joinpath(f"trial_{t}")
        result_root_directory.mkdir(exist_ok=True, parents=True)
        experiment_1(result_root_directory)

        result_root_directory = exp2_result_root_directory.joinpath(f"trial_{t}")
        result_root_directory.mkdir(exist_ok=True, parents=True)
        experiment_2(result_root_directory)

        result_root_directory = exp3_result_root_directory.joinpath(f"trial_{t}")
        result_root_directory.mkdir(exist_ok=True, parents=True)
        experiment_3(result_root_directory)


def plot():
    """
        This functions plots the results
        :return:
        """
    exp1_result_root_directory = Path("results", "experiment_1")
    exp2_result_root_directory = Path("results", "experiment_2")
    exp3_result_root_directory = Path("results", "experiment_3")
    # 20 trial for each experiment
    for t in range(20):
        result_root_directory = exp1_result_root_directory.joinpath(f"trial_{t}")
        result_root_directory.mkdir(exist_ok=True, parents=True)
        experiment_1(result_root_directory)

        result_root_directory = exp2_result_root_directory.joinpath(f"trial_{t}")
        result_root_directory.mkdir(exist_ok=True, parents=True)
        experiment_2(result_root_directory)

        result_root_directory = exp3_result_root_directory.joinpath(f"trial_{t}")
        result_root_directory.mkdir(exist_ok=True, parents=True)
        experiment_3(result_root_directory)


def experiment_1(result_root_directory: Path):
    """
    This function simulates the colony with no cautionary pheromone (with no defending policy)
    :param result_root_directory: where to log the results
    :return:
    """
    config = Config()
    config.POLICY = 0
    config.ANTS = 1024
    config.N = 3
    config.SHOW_HOME_PHERO = True
    config.SHOW_FOOD_PHERO = True
    config.SHOW_MIS_PHERO = True
    config.SHOW_CAUT_PHERO = False
    config.SHOW_COLONY = False

    detractors_fracs = [0.001, 0.002, 0.0039, 0.0078, 0.0156, 0.0313, 0.0625, 0.125, 0.25, 0.50]
    evaporation_rates = [0.0, 0.5, 1., 2., 5., 10., 50., 100., 500., 1000.]

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
    """
    This function simulates the colony using the cautionary pheromone (as in the base-line paper)
    :param result_root_directory: where to log the results
    :return:
    """
    config = Config()
    config.POLICY = 1
    config.ANTS = 1024
    config.N = 3
    config.SHOW_HOME_PHERO = False
    config.SHOW_FOOD_PHERO = True
    config.SHOW_MIS_PHERO = True
    config.SHOW_CAUT_PHERO = True
    config.SHOW_COLONY = False

    detractors_fractions = [0.0313, 0.125]
    evaporation_rates = [1, 5]
    patience_refill_steps = [1, 2, 5, 10, 50, 100, 1000]
    patience_thresholds = [50, 100, 250, 500, 750, 1000]

    for detractors_fraction, evaporation_rate in zip(detractors_fractions, evaporation_rates):
        config.MAL_ANT_FRC = detractors_fraction
        config.K = evaporation_rate
        sub_directory = result_root_directory.joinpath(f"df_{detractors_fraction}_er_{evaporation_rate}")
        for patience_refill_step in patience_refill_steps:
            for patience_threshold in patience_thresholds:
                config.TP = patience_refill_step
                config.P_MAX = patience_threshold

                result_directory = sub_directory.joinpath(f"tp_{patience_refill_step}_pm_{patience_threshold}")
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
    config.ANTS = 1024
    config.N = 3
    config.SHOW_HOME_PHERO = False
    config.SHOW_FOOD_PHERO = True
    config.SHOW_MIS_PHERO = True
    config.SHOW_CAUT_PHERO = True
    config.SHOW_COLONY = False

    detractors_fractions = [0.0313, 0.125]
    evaporation_rates = [1, 5]
    patience_refill_steps = [1, 2, 5, 10, 50, 100, 1000]
    patience_thresholds = [50, 100, 250, 500, 750, 1000]

    for detractors_fraction, evaporation_rate in zip(detractors_fractions, evaporation_rates):
        config.MAL_ANT_FRC = detractors_fraction
        config.K = evaporation_rate
        sub_directory = result_root_directory.joinpath(f"df_{detractors_fraction}_er_{evaporation_rate}")
        for patience_refill_step in patience_refill_steps:
            for patience_threshold in patience_thresholds:
                config.TP = patience_refill_step
                config.P_MAX = patience_threshold

                result_directory = sub_directory.joinpath(f"pr_{patience_refill_step}_pt_{patience_threshold}")
                result_directory.mkdir(exist_ok=True, parents=True)
                collected_foods, delivered_foods, ants_collected_ever, ants_delivered_ever = simulator(config,
                                                                                                       result_directory)

                np.savetxt(result_directory.joinpath("collected_foods.txt"), collected_foods, fmt='%d')
                np.savetxt(result_directory.joinpath("delivered_foods.txt"), delivered_foods, fmt='%d')
                np.savetxt(result_directory.joinpath("ants_collected_ever.txt"), ants_collected_ever, fmt='%d')
                np.savetxt(result_directory.joinpath("ants_delivered_ever.txt"), ants_delivered_ever, fmt='%d')
                with open(str(result_directory) + "\\config.json", 'w') as file:
                    json.dump(config.__dict__, file)


def plot_line_graph_exp1(experiment_root_dir):
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
            cooperator_ants_num = int(config["ANTS"] * (1 - df))
            avg_collected_foods_dict[(df, er)] = collected_foods / cooperator_ants_num
            avg_delivered_foods_dict[(df, er)] = delivered_foods / cooperator_ants_num
            ants_delivered_frac_dict[(df, er)] = ants_delivered / cooperator_ants_num
            ants_collected_frac_dict[(df, er)] = ants_collected / cooperator_ants_num

    configurations = [(0.0313, 1.0), (0.5, 0.0), (0.001, 1000.), (0.125, 5.0)]
    highlight_chars = ['α', 'β', 'γ', 'σ']

    # Plot the confusion matrices
    x = (np.arange(0, config["N"]) / config["N"]) * 100
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
    subdirectories = experiment_root_dir.iterdir()
    for subdirectory in subdirectories:
        experiments_dir = subdirectory.iterdir()
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

        configurations = [(2, 250), (1000, 50), (1, 1000), (50, 500)]
        highlight_chars = ['α', 'β', 'γ', 'σ']

        # Plot the confusion matrices
        x = (np.arange(0, config["N"]) / config["N"]) * 100
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
    subdirectories = experiment_root_dir.iterdir()
    for subdirectory in subdirectories:
        experiments_dir = subdirectory.iterdir()
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

        configurations = [(2, 250), (1000, 50), (1, 1000), (50, 500)]
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
    highlighted_pixels = [(2, 5), (0, 9), (9, 0), (4, 7)]
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
    highlighted_pixels = [(2, 1), (0, 6), (5, 0), (3, 4)]
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


def test():
    config = Config()
    config.FLLSCRN = True  # True for Full screen, or False for Window
    config.ANTS = 256  # Number of Ants in simulation
    config.N = 10000  # number of simulation steps per experiment
    config.WIDTH = 1920  # default 1200
    config.HEIGHT = 1080  # default 800
    config.PRATIO = 4  # Pixel Size for Pheromone grid
    config.L_FOOD = (372, 36)  # food location
    config.R_FOOD = 16  # food source radius
    config.L_NEST = (960, 540)  # nest location
    config.R_NEST = 20  # nest radius

    ##############yper parameters

    config.MAX_SPEED = 50  # maximum ants speed
    config.THETA_MAX = 0.8 * np.pi  # range of vector generation for direction selection
    config.L_MAX = 40  # maximum magnitude of vector generation for direction selection
    config.MU = 0.1 * np.pi  # coefficient for random noise
    config.LAMBDA = 0.01  # coefficient for pheromone intensity
    config.POLICY = 0
    config.K = 1  # Evaporation rate
    config.P_MAX = 1000  # Maximum patience
    config.TP = 5
    config.X = 32  # number of random vectors generated for direction selection
    config.FPS = 60  # 48-90
    config.MAL_ANT_FRC = .03
    config.VSYNC = True  # limit frame rate to refresh rate

#############c olors

    config.HOME_COLOR = (125, 0, 255)
    config.FOOD_COLOR = [0, 255, 0]
    config.HOME_PHEROMONE_COLOR = (0, 0, 255)
    config.FOOD_PHEROMONE_COLOR = (0, 255, 0)
    config.MIS_PHEROMONE_COLOR = (225, 0, 0)
    config.CAUT_PHEROMONE_COLOR = (255, 0, 250)
    config.MAX_PHEROMONE_INTENSITY = 1000

  #################isualization options
    config.SHOW_HOME_PHERO = True
    config.SHOW_FOOD_PHERO = True
    config.SHOW_MIS_PHERO = True
    config.SHOW_CAUT_PHERO = True
    config.SHOW_COLONY = True  # if True, it visualizes the ants as well

    simulator(config)


if __name__ == "__main__":
    main()



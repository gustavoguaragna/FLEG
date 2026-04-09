import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import json
import os

from task import plot_series, plot_by_marker, calculate_times_and_accs

def main():
    parser = argparse.ArgumentParser(description="Recreate figures from FLEG article")

    parser.add_argument("--figure", type=int, help="Figure number to recreate (e.g., 1, 2, 3, etc.)", required=True)

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    exp_root = script_dir / "experiments"

    if not exp_root.exists():
        print(f"ERROR: Directory not found at {exp_root.absolute()}")
        return
    
    files = list(exp_root.glob("*/metrics.json"))

    os.makedirs("./figures", exist_ok=True)

    loaded_dicts = {}

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            file_path_str = str(file)
            
            # Slice the path starting from 'cifar' or 'mnist'
            if "cifar" in file_path_str:
                clean_path = file_path_str[file_path_str.index("cifar"):]
            elif "mnist" in file_path_str:
                clean_path = file_path_str[file_path_str.index("mnist"):]
            else:
                clean_path = file_path_str # Fallback 
                
            # Replace the remaining slash separating the folder and metrics.json with an underscore
            final_key = clean_path.replace("/", "_").replace("\\", "_")
            
            loaded_dicts[final_key] = json.load(f)

    if args.figure == 2:
        plot_series(
            series={
                "FLEG": loaded_dicts["cifar10_Dir01_fedavg_numchunks100_ganepochs30_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "FedAvg": loaded_dicts["cifar10_Dir01_fedavg_baseline_trial2_metrics.json"]["net_acc"],
            },
            series_styles={
                "FLEG": {"color": "navy", "linewidth": 3},
                "FedAvg": {"color": "indianred", "linewidth": 3}
            },
            figsize = (20,4),
            level_markers={
                "N1": 37,
                "N2": 55,
                "N3": 67,
                "N4": 78
            },
            num_xticks=5,

            label_fontsize=22,
            tick_fontsize=18,
            legend_fontsize=20,

            ylabel="Acurácia",
            xlabel="Épocas",

            save=True,
            plot_name="./figures/FLEG_figure2.pdf"
            
        )


    elif args.figure == 3:

        plot_series(
            series={
                "Cifar10 Class FLEG Full": loaded_dicts["cifar10_ClassPartitioner_fedavg_numchunks100_ganepochs30_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Class FLEG Smart": loaded_dicts["cifar10_ClassPartitioner_fedavg_numchunks10_ganepochs35_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Class FLEG Eco": loaded_dicts["cifar10_ClassPartitioner_fedavg_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Class Baseline": loaded_dicts["cifar10_ClassPartitioner_fedavg_baseline_trial1_metrics.json"]["net_acc"],

                "Cifar10 Dir01 FLEG Full": loaded_dicts["cifar10_Dir01_fedavg_numchunks100_ganepochs30_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FLEG Smart": loaded_dicts["cifar10_Dir01_fedavg_numchunks10_ganepochs35_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FLEG Eco": loaded_dicts["cifar10_Dir01_fedavg_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Dir01 Baseline": loaded_dicts["cifar10_Dir01_fedavg_baseline_trial2_metrics.json"]["net_acc"],

                "Cifar10 Dir05 FLEG Full": loaded_dicts["cifar10_Dir05_fedavg_numchunks50_ganepochs25_fixed_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FLEG Smart": loaded_dicts["cifar10_Dir05_fedavg_numchunks10_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FLEG Eco": loaded_dicts["cifar10_Dir05_fedavg_numchunks1_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Dir05 Baseline": loaded_dicts["cifar10_Dir05_fedavg_baseline_trial1_metrics.json"]["net_acc"],


                "Mnist Class FLEG Full": loaded_dicts["mnist_ClassPartitioner_fedavg_numchunks100_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Mnist Class FLEG Smart": loaded_dicts["mnist_ClassPartitioner_fedavg_numchunks10_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Class FLEG Eco": loaded_dicts["mnist_ClassPartitioner_fedavg_numchunks1_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Class Baseline": loaded_dicts["mnist_ClassPartitioner_fedavg_baseline_trial3_metrics.json"]["net_acc"],

                "Mnist Dir01 FLEG Full": loaded_dicts["mnist_Dir01_fedavg_numchunks100_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir01 FLEG Smart": loaded_dicts["mnist_Dir01_fedavg_numchunks10_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Mnist Dir01 FLEG Eco": loaded_dicts["mnist_Dir01_fedavg_numchunks1_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir01 Baseline": loaded_dicts["mnist_Dir01_fedavg_baseline_trial3_metrics.json"]["net_acc"],

                "Mnist Dir05 FLEG Full": loaded_dicts["mnist_Dir05_fedavg_numchunks100_ganepochs20_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir05 FLEG Eco": loaded_dicts["mnist_Dir05_fedavg_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Mnist Dir05 Baseline": loaded_dicts["mnist_Dir05_fedavg_baseline_trial3_metrics.json"]["net_acc"],



                "Cifar10 Class FLEG Full + FedProx": loaded_dicts["cifar10_ClassPartitioner_fedprox_numchunks100_ganepochs30_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Class FLEG Eco + FedProx": loaded_dicts["cifar10_ClassPartitioner_fedprox_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Class FedProx": loaded_dicts["cifar10_ClassPartitioner_fedprox_baseline_trial1_metrics.json"]["net_acc"],

                "Cifar10 Dir01 FLEG Full + FedProx": loaded_dicts["cifar10_Dir01_fedprox_numchunks100_ganepochs30_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FLEG Smart + FedProx": loaded_dicts["cifar10_Dir01_fedprox_numchunks10_ganepochs35_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FLEG Eco + FedProx": loaded_dicts["cifar10_Dir01_fedprox_numchunks1_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FedProx": loaded_dicts["cifar10_Dir01_fedprox_baseline_trial3_metrics.json"]["net_acc"],

                "Cifar10 Dir05 FLEG Full + FedProx": loaded_dicts["cifar10_Dir05_fedprox_numchunks10_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FLEG Eco + FedProx": loaded_dicts["cifar10_Dir05_fedprox_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FedProx": loaded_dicts["cifar10_Dir05_fedprox_baseline_trial3_metrics.json"]["net_acc"],
                "Phantom series": [np.nan],


                "Mnist Class FLEG Full + FedProx": loaded_dicts["mnist_ClassPartitioner_fedprox_numchunks100_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Mnist Class FLEG Smart + FedProx": loaded_dicts["mnist_ClassPartitioner_fedprox_numchunks10_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Mnist Class FLEG Eco + FedProx": loaded_dicts["mnist_ClassPartitioner_fedprox_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Mnist Class FedProx": loaded_dicts["mnist_ClassPartitioner_fedprox_baseline_trial2_metrics.json"]["net_acc"],

                "Mnist Dir01 FLEG Full + FedProx": loaded_dicts["mnist_Dir01_fedprox_numchunks100_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir01 FLEG Smart + FedProx": loaded_dicts["mnist_Dir01_fedprox_numchunks10_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Mnist Dir01 FLEG Eco + FedProx": loaded_dicts["mnist_Dir01_fedprox_numchunks1_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir01 FedProx": loaded_dicts["mnist_Dir01_fedprox_baseline_trial3_metrics.json"]["net_acc"],

                "Mnist Dir05 FLEG Full + FedProx": loaded_dicts["mnist_Dir05_fedprox_numchunks100_ganepochs20_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir05 FLEG Eco + FedProx": loaded_dicts["mnist_Dir05_fedprox_numchunks1_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Mnist Dir05 FedProx": loaded_dicts["mnist_Dir05_fedprox_baseline_trial3_metrics.json"]["net_acc"],



                "Cifar10 Class FLEG Full 3": loaded_dicts["cifar10_ClassPartitioner_fedavg_numchunks100_ganepochs30_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Class FedAvg 3": loaded_dicts["cifar10_ClassPartitioner_fedavg_baseline_trial1_metrics.json"]["net_acc"],
                "Cifar10 Class FLEG Full + FedProx 3": loaded_dicts["cifar10_ClassPartitioner_fedprox_numchunks100_ganepochs30_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Class FedProx 3": loaded_dicts["cifar10_ClassPartitioner_fedprox_baseline_trial1_metrics.json"]["net_acc"],

                "Cifar10 Dir01 FLEG Full 3": loaded_dicts["cifar10_Dir01_fedavg_numchunks100_ganepochs30_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FedAvg 3": loaded_dicts["cifar10_Dir01_fedavg_baseline_trial2_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FLEG Full + FedProx 3": loaded_dicts["cifar10_Dir01_fedprox_numchunks100_ganepochs30_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Cifar10 Dir01 FedProx 3": loaded_dicts["cifar10_Dir01_fedprox_baseline_trial3_metrics.json"]["net_acc"],

                "Cifar10 Dir05 FLEG Full 3": loaded_dicts["cifar10_Dir05_fedavg_numchunks50_ganepochs25_fixed_fleg_trial1_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FedAvg 3": loaded_dicts["cifar10_Dir05_fedavg_baseline_trial1_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FLEG Full + FedProx 3": loaded_dicts["cifar10_Dir05_fedprox_numchunks10_ganepochs25_dynamic_fleg_trial2_metrics.json"]["net_acc"],
                "Cifar10 Dir05 FedProx 3": loaded_dicts["cifar10_Dir05_fedprox_baseline_trial3_metrics.json"]["net_acc"],


                "Mnist Class FLEG Full 3": loaded_dicts["mnist_ClassPartitioner_fedavg_numchunks100_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Mnist Class FedAvg 3": loaded_dicts["mnist_ClassPartitioner_fedavg_baseline_trial3_metrics.json"]["net_acc"],
                "Mnist Class FLEG Full + FedProx 3": loaded_dicts["mnist_ClassPartitioner_fedprox_numchunks100_ganepochs25_dynamic_fleg_trial3_metrics.json"]["net_acc"],
                "Mnist Class FedProx 3": loaded_dicts["mnist_ClassPartitioner_fedprox_baseline_trial2_metrics.json"]["net_acc"],

                "Mnist Dir01 FLEG Full 3": loaded_dicts["mnist_Dir01_fedavg_numchunks100_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir01 FedAvg 3": loaded_dicts["mnist_Dir01_fedavg_baseline_trial3_metrics.json"]["net_acc"],
                "Mnist Dir01 FLEG Full + FedProx 3": loaded_dicts["mnist_Dir01_fedprox_numchunks100_ganepochs25_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir01 FedProx 3": loaded_dicts["mnist_Dir01_fedprox_baseline_trial3_metrics.json"]["net_acc"],

                "Mnist Dir05 FLEG Full 3": loaded_dicts["mnist_Dir05_fedavg_numchunks100_ganepochs20_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir05 FedAvg 3": loaded_dicts["mnist_Dir05_fedavg_baseline_trial3_metrics.json"]["net_acc"],
                "Mnist Dir05 FLEG Full + FedProx 3": loaded_dicts["mnist_Dir05_fedprox_numchunks100_ganepochs20_dynamic_fleg_trial1_metrics.json"]["net_acc"],
                "Mnist Dir05 FedProx 3": loaded_dicts["mnist_Dir05_fedprox_baseline_trial3_metrics.json"]["net_acc"],

            },

            subplot_groups = [
                ["Cifar10 Class FLEG Full", "Cifar10 Class FLEG Smart", "Cifar10 Class FLEG Eco", "Cifar10 Class Baseline"],
                ["Cifar10 Dir01 FLEG Full", "Cifar10 Dir01 FLEG Smart", "Cifar10 Dir01 FLEG Eco", "Cifar10 Dir01 Baseline"],
                ["Cifar10 Dir05 FLEG Full", "Cifar10 Dir05 FLEG Smart", "Cifar10 Dir05 FLEG Eco", "Cifar10 Dir05 Baseline"],

                ["Mnist Class FLEG Full", "Mnist Class FLEG Smart", "Mnist Class FLEG Eco", "Mnist Class Baseline"],
                ["Mnist Dir01 FLEG Full", "Mnist Dir01 FLEG Smart", "Mnist Dir01 FLEG Eco", "Mnist Dir01 Baseline"],
                ["Mnist Dir05 FLEG Full", "Mnist Dir05 FLEG Eco", "Mnist Dir05 Baseline"],


                ["Cifar10 Class FLEG Full + FedProx", "Cifar10 Class FLEG Eco + FedProx", "Cifar10 Class FedProx"],
                ["Cifar10 Dir01 FLEG Full + FedProx", "Cifar10 Dir01 FLEG Smart + FedProx", "Cifar10 Dir01 FLEG Eco + FedProx", "Cifar10 Dir01 FedProx"],
                ["Cifar10 Dir05 FLEG Full + FedProx", "Cifar10 Dir05 FLEG Eco + FedProx", "Cifar10 Dir05 FedProx", "Phantom series"],

                ["Mnist Class FLEG Full + FedProx", "Mnist Class FLEG Smart + FedProx", "Mnist Class FLEG Eco + FedProx", "Mnist Class FedProx"],
                ["Mnist Dir01 FLEG Full + FedProx", "Mnist Dir01 FLEG Smart + FedProx", "Mnist Dir01 FLEG Eco + FedProx", "Mnist Dir01 FedProx"],
                ["Mnist Dir05 FLEG Full + FedProx", "Mnist Dir05 FLEG Eco + FedProx", "Mnist Dir05 FedProx"],


                ["Cifar10 Class FLEG Full 3", "Cifar10 Class FedAvg 3", "Cifar10 Class FLEG Full + FedProx 3", "Cifar10 Class FedProx 3"],
                ["Cifar10 Dir01 FLEG Full 3", "Cifar10 Dir01 FedAvg 3", "Cifar10 Dir01 FLEG Full + FedProx 3", "Cifar10 Dir01 FedProx 3"],
                ["Cifar10 Dir05 FLEG Full 3", "Cifar10 Dir05 FedAvg 3", "Cifar10 Dir05 FLEG Full + FedProx 3", "Cifar10 Dir05 FedProx 3"],

                ["Mnist Class FLEG Full 3", "Mnist Class FedAvg 3", "Mnist Class FLEG Full + FedProx 3", "Mnist Class FedProx 3"],
                ["Mnist Dir01 FLEG Full 3", "Mnist Dir01 FedAvg 3", "Mnist Dir01 FLEG Full + FedProx 3", "Mnist Dir01 FedProx 3"],
                ["Mnist Dir05 FLEG Full 3", "Mnist Dir05 FedAvg 3", "Mnist Dir05 FLEG Full + FedProx 3", "Mnist Dir05 FedProx 3"],
                
            ],
            series_styles={
                "Cifar10 Class FLEG Full": {"color": "navy", "label": "FLEG Full"},
                "Cifar10 Class FLEG Smart": {"color": "cornflowerblue", "label": "FLEG Smart"},
                "Cifar10 Class FLEG Eco": {"color": "deepskyblue", "label": "FLEG Eco"},
                "Cifar10 Class Baseline": {"color": "indianred", "label": "FedAvg"},

                "Cifar10 Dir01 FLEG Full": {"color": "navy", "label": "FLEG Full"},
                "Cifar10 Dir01 FLEG Smart": {"color": "cornflowerblue", "label": "FLEG Smart"},
                "Cifar10 Dir01 FLEG Eco": {"color": "deepskyblue", "label": "FLEG Eco"},
                "Cifar10 Dir01 Baseline": {"color": "indianred", "label": "FedAvg"},

                "Cifar10 Dir05 FLEG Full": {"color": "navy", "label": "FLEG Full"},
                "Cifar10 Dir05 FLEG Smart": {"color": "cornflowerblue", "label": "FLEG Smart"},
                "Cifar10 Dir05 FLEG Eco": {"color": "deepskyblue", "label": "FLEG Eco"},
                "Cifar10 Dir05 Baseline": {"color": "indianred", "label": "FedAvg"},


                "Mnist Class FLEG Full": {"color": "navy", "label": "FLEG Full"},
                "Mnist Class FLEG Smart":{"color": "cornflowerblue", "label": "FLEG Smart"},
                "Mnist Class FLEG Eco": {"color": "deepskyblue", "label": "FLEG Eco"},
                "Mnist Class Baseline": {"color": "indianred", "label": "FedAvg"},

                "Mnist Dir01 FLEG Full": {"color": "navy", "label": "FLEG Full"},
                "Mnist Dir01 FLEG Smart":{"color": "cornflowerblue", "label": "FLEG Smart"},
                "Mnist Dir01 FLEG Eco": {"color": "deepskyblue", "label": "FLEG Eco"},
                "Mnist Dir01 Baseline": {"color": "indianred", "label": "FedAvg"},

                "Mnist Dir05 FLEG Full": {"color": "navy", "label": "FLEG Full"},
                "Mnist Dir05 FLEG Eco": {"color": "deepskyblue", "label": "FLEG Eco"},
                "Mnist Dir05 Baseline": {"color": "indianred", "label": "FedAvg"},



                "Cifar10 Class FLEG Full + FedProx": {"color": "darkolivegreen", "label": "FLEG Full + FedProx"},
                "Cifar10 Class FLEG Eco + FedProx": {"color": "lightgreen", "label": "FLEG Eco + FedProx"},
                "Cifar10 Class FedProx": {"color": "goldenrod", "label": "FedProx"},

                "Cifar10 Dir01 FLEG Full + FedProx": {"color": "darkolivegreen", "label": "FLEG Full + FedProx"},
                "Cifar10 Dir01 FLEG Smart + FedProx": {"color": "forestgreen", "label": "FLEG Smart + FedProx"},
                "Cifar10 Dir01 FLEG Eco + FedProx": {"color": "lightgreen", "label": "FLEG Eco + FedProx"},
                "Cifar10 Dir01 FedProx": {"color": "goldenrod", "label": "FedProx"},

                "Cifar10 Dir05 FLEG Full + FedProx": {"color": "darkolivegreen", "label": "FLEG Full + FedProx"},
                "Cifar10 Dir05 FLEG Eco + FedProx": {"color": "lightgreen", "label": "FLEG Eco + FedProx"},
                "Cifar10 Dir05 FedProx": {"color": "goldenrod", "label": "FedProx"},
                "Phantom series": {"color": "forestgreen", "label": "FLEG Smart + FedProx"},


                "Mnist Class FLEG Full + FedProx": {"color": "darkolivegreen", "label": "FLEG Full + FedProx"},
                "Mnist Class FLEG Smart + FedProx":{"color": "forestgreen", "label": "FLEG Smart + FedProx"},
                "Mnist Class FLEG Eco + FedProx": {"color": "lightgreen", "label": "FLEG Eco + FedProx"},
                "Mnist Class FedProx": {"color": "goldenrod", "label": "FedProx"},

                "Mnist Dir01 FLEG Full + FedProx": {"color": "darkolivegreen", "label": "FLEG Full + FedProx"},
                "Mnist Dir01 FLEG Smart + FedProx":{"color": "forestgreen", "label": "FLEG Smart + FedProx"},
                "Mnist Dir01 FLEG Eco + FedProx": {"color": "lightgreen", "label": "FLEG Eco + FedProx"},
                "Mnist Dir01 FedProx": {"color": "goldenrod", "label": "FedProx"},

                "Mnist Dir05 FLEG Full + FedProx": {"color": "darkolivegreen", "label": "FLEG Full + FedProx"},
                "Mnist Dir05 FLEG Eco + FedProx": {"color": "lightgreen", "label": "FLEG Eco + FedProx"},
                "Mnist Dir05 FedProx": {"color": "goldenrod", "label": "FedProx"},



                "Cifar10 Class FLEG Full 3": {"color": "navy", "label": "FLEG"},
                "Cifar10 Class FedAvg 3": {"color": "indianred", "label": "FedAvg"},
                "Cifar10 Class FLEG Full + FedProx 3": {"color": "darkolivegreen", "label": "FLEG + FedProx"},
                "Cifar10 Class FedProx 3": {"color": "goldenrod", "label": "FedProx"},

                "Cifar10 Dir01 FLEG Full 3": {"color": "navy", "label": "FLEG"},
                "Cifar10 Dir01 FedAvg 3": {"color": "indianred", "label": "FedAvg"},
                "Cifar10 Dir01 FLEG Full + FedProx 3": {"color": "darkolivegreen", "label": "FLEG + FedProx"},
                "Cifar10 Dir01 FedProx 3": {"color": "goldenrod", "label": "FedProx"},

                "Cifar10 Dir05 FLEG Full 3": {"color": "navy", "label": "FLEG"},
                "Cifar10 Dir05 FedAvg 3": {"color": "indianred", "label": "FedAvg"},
                "Cifar10 Dir05 FLEG Full + FedProx 3": {"color": "darkolivegreen", "label": "FLEG + FedProx"},
                "Cifar10 Dir05 FedProx 3": {"color": "goldenrod", "label": "FedProx"},


                "Mnist Class FLEG Full 3": {"color": "navy", "label": "FLEG"},
                "Mnist Class FedAvg 3": {"color": "indianred", "label": "FedAvg"},
                "Mnist Class FLEG Full + FedProx 3": {"color": "darkolivegreen", "label": "FLEG + FedProx"},
                "Mnist Class FedProx 3": {"color": "goldenrod", "label": "FedProx"},

                "Mnist Dir01 FLEG Full 3": {"color": "navy", "label": "FLEG"},
                "Mnist Dir01 FedAvg 3": {"color": "indianred", "label": "FedAvg"},
                "Mnist Dir01 FLEG Full + FedProx 3": {"color": "darkolivegreen", "label": "FLEG + FedProx"},
                "Mnist Dir01 FedProx 3": {"color": "goldenrod", "label": "FedProx"},

                "Mnist Dir05 FLEG Full 3": {"color": "navy", "label": "FLEG"},
                "Mnist Dir05 FedAvg 3": {"color": "indianred", "label": "FedAvg"},
                "Mnist Dir05 FLEG Full + FedProx 3": {"color": "darkolivegreen", "label": "FLEG + FedProx"},
                "Mnist Dir05 FedProx 3": {"color": "goldenrod", "label": "FedProx"},
                
            },

            subplot_layout=(6,3),
            figsize=(20,18),

            title=["ClassPartition", "Dir01", "Dir05", "","",""]*3,
            title_fontsize=17,

            row_suptitles=[
                "a) Variações de FLEG","",
                "b) Variações de FLEG + FedProx", "",
                "c) FLEG x Baselines", ""
            ],
            row_suptitle_fontsize=18,
            subplot_margins={'top': 0.96, 'bottom': -0.06, 'left': 0.049, 'right': 0.969},
            hspace=0.75,

            row_labels=["CIFAR10", "MNIST"]*3,
            row_label_fontsize=15,

            xlabel=["","","",
                    "Épocas", "Épocas", "Épocas"]*3,
            ylabel="Acurácia",
            label_fontsize=16,

            tick_fontsize=15,

            ylim=[(0.1, 0.4),(0., 0.45),(0.1, 0.5), (0.2, 1.),(0.7, 1.),(0.9, 1.),  (0.1, 0.4),(0., 0.45),(0.1, 0.5), (0.2, 1.),(0.7, 1.),(0.9, 1.),  (0.1, 0.4),(0, 0.45),(0.1, 0.5), (0.2,1.),(0.7,1.),(0.9, 1.)],
            num_yticks=3,

            xlim=[(1, 100),(1, 100),(1, 150), (1, 100),(1, 150),(1, 100),  (1, 200),(1, 200),(1, 250), (1, 200),(1, 200),(1, 200),  (1, 200),(1, 200),(1, 250), (1, 200),(1, 200),(1, 200)],
            x_ticks=[list(range(0, 101, 20)), list(range(0, 101, 20)), list(range(0, 151, 30)),
                    list(range(0, 101, 20)), list(range(0, 151, 30)), list(range(0, 101, 20)),
                    
                    list(range(0, 201, 40)), list(range(0, 201, 40)), list(range(0, 251, 50)),
                    list(range(0, 201, 40)), list(range(0, 201, 40)), list(range(0, 201, 40)),
                    
                    list(range(0, 201, 40)), list(range(0, 201, 40)), list(range(0, 251, 50)),
                    list(range(0, 201, 40)), list(range(0, 201, 40)), list(range(0, 201, 40))],

            legend_fontsize=18,
            legend_subplot_index=[2,8,14],
            legend_loc="lower center",
            legend_kwargs={"frameon": False, "bbox_to_anchor": (0.7, 0.03),"ncol": 1, "borderaxespad": 0.0001,
                            "columnspacing": 0.7,"handlelength": 1, "labelspacing": 0.1,"handletextpad": 0.2},
            
            save=True,
            plot_name="./figures/FLEG_figure3.pdf",
        )
    
    elif args.figure == 4:
        baseline_values = {}

        for key, data in loaded_dicts.items():
            if 'baseline' in key:
                net_acc_length = len(data.get('net_acc', []))
                
                if 'cifar10' in key:
                    value = min(150, net_acc_length) * 0.25 * 2
                elif 'mnist' in key:
                    value = min(150, net_acc_length) * 0.18 * 2
                
                baseline_values[key] = value
            
        gb_cifar = []
        acc_cifar = []
        cores_cifar = []
        markers_cifar = []
        gb_mnist = []
        acc_mnist = []
        cores_mnist = []
        markers_mnist = []

        for exp_name, exp_dict in loaded_dicts.items():
            if "fedprox" in exp_name:
                continue

            if "fleg" in exp_name:

                if "cifar" in exp_name:
                    gb_cifar.append(sum(exp_dict["MB_transmission"])/1e3)
                    acc_cifar.append(max(exp_dict["net_acc"]))

                    if "Class" in exp_name:
                        cores_cifar.append("firebrick")
                    elif "Dir01" in exp_name:
                        cores_cifar.append("gold")
                    elif "Dir05" in exp_name:
                        cores_cifar.append("limegreen")
                    else:
                        raise ValueError(f"{exp_name} with no partition")

                    if "numchunks100" in exp_name or "numchunks50" in exp_name:
                        markers_cifar.append("$F$")
                    elif "numchunks10_" in exp_name:
                        markers_cifar.append("$S$")
                    elif "numchunks1_" in exp_name:
                        markers_cifar.append("$E$")
                    else:
                        raise ValueError(f"{exp_name} with no mode")
                    
                elif "mnist" in exp_name:
                    gb_mnist.append(sum(exp_dict["MB_transmission"])/1e3)
                    acc_mnist.append(max(exp_dict["net_acc"]))
                    
                    if "Class" in exp_name:
                        cores_mnist.append("firebrick")
                    elif "Dir01" in exp_name:
                        cores_mnist.append("gold")
                    elif "Dir05" in exp_name:
                        cores_mnist.append("limegreen")
                    else:
                        raise ValueError(f"{exp_name} with no partition")

                    if "numchunks100" in exp_name:
                        markers_mnist.append("$F$")
                    elif "numchunks10_" in exp_name:
                        markers_mnist.append("$S$")
                    elif "numchunks1_" in exp_name:
                        markers_mnist.append("$E$")
                    else:
                        raise ValueError(f"{exp_name} with no mode")
                
                else:
                    raise ValueError(f"{exp_name} with no dataset")

            elif "baseline" in exp_name:
                if "cifar" in exp_name:
                    gb_cifar.append(baseline_values[exp_name]/1e3)
                    acc_cifar.append(max(exp_dict["net_acc"][:150]))
                    markers_cifar.append("o") 
                    if "Class" in exp_name:
                        cores_cifar.append("firebrick")
                    elif "Dir01" in exp_name:
                        cores_cifar.append("gold")
                    elif "Dir05" in exp_name:
                        cores_cifar.append("limegreen")
                    else:
                        raise ValueError(f"{exp_name} with no partition")

                elif "mnist" in exp_name:
                    gb_mnist.append(baseline_values[exp_name]/1e3)
                    acc_mnist.append(max(exp_dict["net_acc"][:150]))
                    markers_mnist.append("o")
                    if "Class" in exp_name:
                        cores_mnist.append("firebrick")
                    elif "Dir01" in exp_name:
                        cores_mnist.append("gold")
                    elif "Dir05" in exp_name:
                        cores_mnist.append("limegreen")
                    else:
                        raise ValueError(f"{exp_name} with no partition")
                
                else:
                    raise ValueError(f"{exp_name} with no dataset")
                
            else:
                raise ValueError(f"{exp_name} is fleg or baseline?")
            

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,4)) 

        plot_by_marker(ax1, acc_cifar, gb_cifar, cores_cifar, markers_cifar, "a) CIFAR-10")
        plot_by_marker(ax2, acc_mnist, gb_mnist, cores_mnist, markers_mnist, "b) MNIST")

        legend_colors = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', label='Dir01', markersize=16),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', label='Dir05', markersize=16),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='firebrick', label='Class', markersize=16)
        ]

        legend_markers = [
            Line2D([0], [0], marker='o', color='w', markeredgecolor='k', markerfacecolor='gray', label='FedAvg', markersize=16),
            Line2D([0], [0], marker='$F$', color='w', markeredgecolor='k', markerfacecolor='k', label='FLEG Full', markersize=16),
            Line2D([0], [0], marker='$S$', color='w', markeredgecolor='k', markerfacecolor='k', label='FLEG Smart', markersize=16),
            Line2D([0], [0], marker='$E$', color='w', markeredgecolor='k', markerfacecolor='k', label='FLEG Eco', markersize=16)
        ]

        plt.subplots_adjust(right=0.99, wspace=0.2, bottom=0.2, top=0.9, left=0.06)

        l1 = fig.legend(handles=legend_colors, title="Partição (Cores)", title_fontsize=16, loc='center left', 
                        bbox_to_anchor=(0.99, 0.75), frameon=False, fontsize=16)

        l2 = fig.legend(handles=legend_markers, title="Método (Marcadores)", title_fontsize=16, loc='center left', 
                        bbox_to_anchor=(0.99, 0.35), frameon=False, fontsize=16)

        fig.add_artist(l1)

        plt.tight_layout()
        plt.savefig("./figures/FLEG_figure4.pdf", bbox_inches='tight')
        plt.show()

    elif args.figure == 5:
        # 1. Regex Pattern
        pattern = re.compile(
            r"(cifar10|mnist)_" 
            r"(ClassPartitioner|Dir\d{2})_"
            r"(fedprox|fedavg)_"
            r"(?:numchunks(\d+)_|)"
            r"(?:ganepochs(\d+)_|)"
            r"(fixed_|dynamic_|)"
            r"(fleg|baseline)_"
            r"trial(\d+)"
        )

        ylim_settings = {
            "cifar10ClassPartitioner": (0.1, 0.4), 
            "cifar10Dir01": (0.1, 0.6),
            "cifar10Dir05": (0.1, 0.6),
            "mnistClassPartitioner": (0.2, 1),
            "mnistDir01": (0.7, 1.),
            "mnistDir05": (0.9, 1)
        }

        # 3. Plotting Setup
        fig, axes = plt.subplots(2, 3, figsize=(20, 5), constrained_layout=True)
        datasets = ["cifar10", "mnist"]
        partitions = ["ClassPartition", "Dir01", "Dir05"]
        unique_legend_items = {}

        # 4. Main Loop
        for i, dataset in enumerate(datasets):
            for j, partition in enumerate(partitions):
                ax = axes[i, j]

                if j in [1,2]:
                    ax.set_ylim(ylim_settings[f"{dataset}{partition}"])
                else:
                    ax.set_ylim(ylim_settings[f"{dataset}{partition}er"])
                
                ax.set_xlim((0, 100))

                ax.tick_params(axis='y', labelsize=14)
                ax.tick_params(axis='x', labelsize=14)
                
                # Filter keys
                subplot_keys = []
                for key in loaded_dicts:
                    match = pattern.match(key)
                    if match:
                        if j in [1,2]:
                            if match.group(1) == dataset and match.group(2) == partition:
                                subplot_keys.append(key)
                        else:
                            if match.group(1) == dataset and match.group(2) == f"{partition}er":
                                subplot_keys.append(key)
                
                # Sort keys (Baseline first)
                subplot_keys.sort(key=lambda x: (
                    0 if "baseline" in x else 1,
                    int(pattern.match(x).group(4)) if not "baseline" in x else None ,
                    int(pattern.match(x).group(8))
                ))
                
                # Plot Lines
                for key in subplot_keys:
                    match = pattern.match(key)
                    exp_type = match.group(7)
                    is_baseline = (exp_type == 'baseline')
                    
                    # Extract data using the helper
                    times, accs = calculate_times_and_accs(loaded_dicts[key], is_baseline)
                    times_min = [t / 60.0 for t in times]

                    exp_base = match.group(3) 
                    
                    # Generate Label
                    if is_baseline:
                        if exp_base == "fedavg":
                            label = "FedAvg"
                            color = "indianred"
                        elif exp_base == "fedprox":
                            label = "FedProx"
                            color = "goldenrod"
                        else:
                            raise ValueError(f"{exp_base} is not fedavg nor fedprox")
                    else:
                        chunks = int(match.group(4))
                        gan_epoch = int(match.group(5))
                        if exp_base == "fedavg":
                            if chunks == 1:
                                    label = "FLEG Eco"
                                    color = "deepskyblue"
                            elif chunks == 10:
                                label = "FLEG Smart"
                                color = "cornflowerblue"
                            else:
                                label = "FLEG Full"
                                color = "navy"
                        elif exp_base == "fedprox":
                            if chunks == 1:
                                    label = "FLEG Eco + FedProx"
                                    color = "lightgreen"
                            elif chunks == 10:
                                label = "FLEG Smart + FedProx"
                                color = "forestgreen"
                            else:
                                label = "FLEG Full + FedProx"
                                color = "darkolivegreen"
                    
                    
                    line, = ax.plot(times_min, accs, label=label, color=color)

                    unique_legend_items[label] = line

                # Formatting
                if dataset == "cifar10":
                    ax.set_title(f"{partition}", fontsize=18)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                if i == 1: # Bottom row
                    ax.set_xlabel("Tempo (minutos)", fontsize=16)
                if j == 0: # Left column
                    ax.set_ylabel("Acurácia", fontsize=16)

                if j == 2:
                    ax.text(
                            1.01, 0.5, dataset.upper(),  # x=1.05 (slightly outside right), y=0.5 (center)
                            transform=ax.transAxes,      # Coordinates relative to the subplot
                            rotation=270,                # Vertical rotation
                            ha='left', 
                            va='center',
                            fontsize=14,
                            fontweight='bold'            # Optional: make it bold to distinguish from data
                        )

        label_order = ["FLEG Full", "FLEG Smart", "FLEG Eco", "FedAvg", "FLEG Full + FedProx", "FLEG Smart + FedProx", "FLEG Eco + FedProx", "FedProx"]
        sorted_labels = sorted(unique_legend_items.keys(), key=lambda x: label_order.index(x) if x in label_order else 999)
        sorted_handles = [unique_legend_items[l] for l in sorted_labels]
        fig.legend(sorted_handles, sorted_labels, loc='lower right', bbox_to_anchor=(0.99, 0.1), fontsize=13, ncols=2, frameon=False,
                columnspacing=1., handlelength=1.8, labelspacing=0.4, handletextpad=0.7)
        plt.savefig("./figures/FLEG_figure5.pdf")
        plt.show()

    else:
        raise ValueError(f"Figure {args.figure} not implemented")


if __name__ == "__main__":    main()
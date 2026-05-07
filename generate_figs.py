import re
import argparse
import math
from pathlib import Path
import json


DEFAULT_FIGURE2_SERIES = [
    ("FLEG", "cifar10_Dir01_fedavg_numchunks100_ganepochs30_dynamic_fleg_trial3_metrics.json"),
    ("FedAvg", "cifar10_Dir01_fedavg_baseline_trial2_metrics.json"),
]

DEFAULT_FIGURE2_STYLES = {
    "FLEG": {"color": "navy", "linewidth": 3},
    "FedAvg": {"color": "indianred", "linewidth": 3},
}

DEFAULT_BASELINE_MB_PER_EPOCH = {
    "cifar": 0.25,
    "mnist": 0.18,
}

DEFAULT_BASELINE_TRAFFIC_FACTOR = 2


def parse_name_value(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use o formato nome=valor.")
    name, raw_value = value.split("=", 1)
    name = name.strip()
    raw_value = raw_value.strip()
    if not name or not raw_value:
        raise argparse.ArgumentTypeError("Nome e valor precisam ser preenchidos.")
    return name, raw_value


def parse_level_markers(value):
    markers = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        label, raw_position = parse_name_value(item)
        try:
            markers[label] = int(raw_position)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Marcadores devem usar posições inteiras.") from exc
    return markers


def parse_figsize(value):
    normalized = value.lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Use o formato largura,altura. Exemplo: 20,4.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("As dimensões da figura devem ser numéricas.") from exc


def parse_arguments():
    parser = argparse.ArgumentParser(description="Gera figuras a partir de arquivos metrics.json do FLEG.")

    parser.add_argument("--figure", type=int, choices=[2, 3, 4, 5], help="Número da figura a gerar.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Diretório que contém subpastas de experimentos. Padrão: ./paper_experiments.",
    )
    parser.add_argument(
        "--metrics-file",
        default="metrics.json",
        help="Nome do arquivo de métricas dentro de cada experimento.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Diretório onde as figuras serão salvas.",
    )
    parser.add_argument("--output-prefix", default="FLEG", help="Prefixo dos arquivos de saída.")
    parser.add_argument(
        "--output-format",
        default="png",
        choices=["pdf", "png", "svg"],
        help="Formato das figuras salvas.",
    )
    parser.add_argument(
        "--metric-label",
        default="Acurácia",
        help="Rótulo exibido para a métrica principal.",
    )
    parser.add_argument(
        "--traffic-key",
        default="MB_transmission",
        help="Chave usada para tráfego nas figuras que calculam custo de comunicação.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Lista as chaves de experimentos encontradas e encerra.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Salva a figura sem abrir uma janela interativa.",
    )

    parser.add_argument("--linewidth", type=float, default=3, help="Espessura das linhas da Figura 2.")
    parser.add_argument(
        "--level-markers",
        type=parse_level_markers,
        default=None,
        help="Marcadores verticais da Figura 2. Exemplo: N1=37,N2=55.",
    )
    parser.add_argument(
        "--no-level-markers",
        action="store_true",
        help="Remove os marcadores verticais da Figura 2.",
    )
    parser.add_argument(
        "--figure2-figsize",
        type=parse_figsize,
        default=(20, 4),
        help="Tamanho da Figura 2 no formato largura,altura.",
    )

    parser.add_argument(
        "--baseline-max-epochs",
        type=int,
        default=150,
        help="Número máximo de épocas usado para estimar tráfego de baselines.",
    )
    args = parser.parse_args()
    args.output_format = args.output_format.lstrip(".")

    if args.figure is None and not args.list_experiments:
        parser.error("Informe --figure ou use --list-experiments.")

    return args


def metrics_key_from_path(metrics_path, experiments_dir):
    relative_path = metrics_path.relative_to(experiments_dir)
    return "_".join(relative_path.parts)


def load_metrics(experiments_dir, metrics_file):
    if not experiments_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {experiments_dir.absolute()}")

    files = sorted(experiments_dir.glob(f"*/{metrics_file}"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo {metrics_file!r} encontrado em {experiments_dir.absolute()}"
        )

    loaded_dicts = {}
    for metrics_path in files:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        loaded_dicts[metrics_key_from_path(metrics_path, experiments_dir)] = metrics
    return loaded_dicts


def resolve_experiment_key(loaded_dicts, reference, metrics_file):
    path_key = "_".join(Path(reference).parts)
    candidates = [reference, path_key]
    if not path_key.endswith(metrics_file):
        candidates.append(f"{path_key}_{metrics_file}")

    for candidate in candidates:
        if candidate in loaded_dicts:
            return candidate

    matches = []
    for key in loaded_dicts:
        if any(key.endswith(f"_{candidate}") for candidate in candidates):
            matches.append(key)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Referência ambígua {reference!r}. Possibilidades: {', '.join(matches)}")

    sample = ", ".join(list(loaded_dicts.keys())[:5])
    raise KeyError(f"Experimento {reference!r} não encontrado. Exemplos disponíveis: {sample}")


def get_metric(loaded_dicts, experiment_reference, metrics_file):
    experiment_key = resolve_experiment_key(loaded_dicts, experiment_reference, metrics_file)
    metrics = loaded_dicts[experiment_key]
    if "net_acc" not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"Métrica 'net_acc' ausente em {experiment_key}. Disponíveis: {available}")
    return metrics["net_acc"]


def build_figure2_series(loaded_dicts, args):
    series = {}
    for label, experiment_reference in DEFAULT_FIGURE2_SERIES:
        series[label] = get_metric(loaded_dicts, experiment_reference, args.metrics_file)
    return series


def build_level_markers_from_accuracy_transition(loaded_dicts, args):
    for _, experiment_reference in DEFAULT_FIGURE2_SERIES:
        experiment_key = resolve_experiment_key(loaded_dicts, experiment_reference, args.metrics_file)
        metrics = loaded_dicts[experiment_key]
        transitions = metrics.get("accuracy_transition")
        net_acc = metrics.get("net_acc")

        if not transitions or not net_acc:
            continue

        markers = {}
        # O último valor marca o fim do treinamento; as linhas verticais indicam transições entre níveis.
        for level_index, target_acc in enumerate(transitions[:-1], start=1):
            marker_position = None
            for epoch_index, acc in enumerate(net_acc):
                if math.isclose(acc, target_acc, rel_tol=1e-9, abs_tol=1e-12):
                    marker_position = epoch_index
                    break

            if marker_position is not None:
                markers[f"N{level_index}"] = marker_position

        if markers:
            return markers

    return None


def build_figure2_styles(series_names, args):
    styles = {}
    for name in series_names:
        style = DEFAULT_FIGURE2_STYLES.get(name, {}).copy()
        style["linewidth"] = args.linewidth
        styles[name] = style
    return styles


def figure_path(args, figure_number):
    return args.output_dir / f"{args.output_prefix}_figure{figure_number}.{args.output_format}"


def estimate_baseline_traffic_mb(exp_name, exp_dict, args):
    if args.traffic_key in exp_dict:
        return sum(exp_dict[args.traffic_key])

    net_acc_length = len(exp_dict.get("net_acc", []))
    epochs = min(args.baseline_max_epochs, net_acc_length)

    if "cifar" in exp_name:
        mb_per_epoch = DEFAULT_BASELINE_MB_PER_EPOCH["cifar"]
    elif "mnist" in exp_name:
        mb_per_epoch = DEFAULT_BASELINE_MB_PER_EPOCH["mnist"]
    else:
        raise ValueError(
            f"{exp_name} não possui {args.traffic_key!r}; informe o tráfego no metrics.json."
        )

    return epochs * mb_per_epoch * DEFAULT_BASELINE_TRAFFIC_FACTOR


def main():
    args = parse_arguments()

    script_dir = Path(__file__).parent
    exp_root = args.experiments_dir or script_dir / "paper_experiments"
    exp_root = exp_root.expanduser().resolve()

    try:
        loaded_dicts = load_metrics(exp_root, args.metrics_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    if args.list_experiments:
        print(f"Experimentos encontrados em {exp_root}:")
        for key in loaded_dicts:
            print(key)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plot_utils import plot_series, plot_by_marker, calculate_times_and_accs

    if args.figure == 2:
        series = build_figure2_series(loaded_dicts, args)
        level_markers = None
        if not args.no_level_markers:
            level_markers = args.level_markers or build_level_markers_from_accuracy_transition(loaded_dicts, args)

        plot_series(
            series=series,
            series_styles=build_figure2_styles(series.keys(), args),
            figsize=args.figure2_figsize,
            level_markers=level_markers,
            num_xticks=5,

            label_fontsize=22,
            tick_fontsize=18,
            legend_fontsize=20,

            ylabel=args.metric_label,
            xlabel="Épocas",

            save=True,
            plot_name=str(figure_path(args, 2)),
            show=not args.no_show
            
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
            ylabel=args.metric_label,
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
            plot_name=str(figure_path(args, 3)),
            show=not args.no_show,
        )
    
    elif args.figure == 4:
        baseline_values = {}

        for key, data in loaded_dicts.items():
            if 'baseline' in key:
                baseline_values[key] = estimate_baseline_traffic_mb(key, data, args)
            
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
                    gb_cifar.append(sum(exp_dict[args.traffic_key])/1e3)
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
                    gb_mnist.append(sum(exp_dict[args.traffic_key])/1e3)
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
                    acc_cifar.append(max(exp_dict["net_acc"][:args.baseline_max_epochs]))
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
                    acc_mnist.append(max(exp_dict["net_acc"][:args.baseline_max_epochs]))
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
        ax1.set_xlabel(f"{args.metric_label} Máxima", fontsize=20)
        ax2.set_xlabel(f"{args.metric_label} Máxima", fontsize=20)

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
        plt.savefig(figure_path(args, 4), bbox_inches='tight')
        if args.no_show:
            plt.close(fig)
        else:
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
                    ax.set_ylabel(args.metric_label, fontsize=16)

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
        plt.savefig(figure_path(args, 5))
        if args.no_show:
            plt.close(fig)
        else:
            plt.show()

    else:
        raise ValueError(f"Figure {args.figure} not implemented")


if __name__ == "__main__":    main()

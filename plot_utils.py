import math
from typing import Any, Iterable, List, Literal, Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


def plot_series(
    series: Mapping[str, Iterable[float]],
    *,
    series_styles: Mapping[str, Mapping[str, Any]] = None,
    subplot_groups: List[List[str]] = None,
    subplot_layout: Tuple[int, int] = None,
    subplot_margins: dict = None,
    legend_subplot_index: Union[int, List[int], str] = "all",
    legend_loc: str = "best",
    legend_fontsize: float = 10,
    legend_kwargs: Union[Mapping[str, Any], List[Mapping[str, Any]]] = None,
    title: Union[str, List[str]] = None,
    title_fontsize: float = None,
    row_suptitles: List[str] = None,
    row_suptitle_fontsize: float = 14,
    figure_title: str = None,
    figure_title_fontsize: float = 16,
    figure_title_y: float = None,
    x_ticks: Union[List[float], List[List[float]]] = None,
    y_ticks: Union[List[float], List[List[float]]] = None,
    xtick_step: Union[int, List[int]] = 1,
    xtick_offset: int = 0,
    first_step_xtick: Union[int, List[int]] = None,
    tick_fontsize: float = None,
    num_xticks: Union[int, List[int]] = None,
    num_yticks: Union[int, List[int]] = None,
    hide_inner_ticks: bool = False,
    xlim: Union[tuple[float, float], List[tuple[float, float]]] = None,
    ylim: Union[tuple[float, float], List[tuple[float, float]]] = None,
    xlabel: Union[str, List[str]] = "Épocas",
    ylabel: Union[str, List[str]] = "Valor",
    label_fontsize: float = None,
    row_labels: List[str] = None,
    row_label_fontsize: float = None,
    highlight: Mapping[str, Literal["max", "min", "both"]] = None,
    highlight_marker: str = "o",
    highlight_markersize: float = 4,
    highlight_color: str = None,
    highlight_text_size: int = 8,
    highlight_text_color: str = None,
    highlight_arrow_color: str = None,
    highlight_arrow_style: str = "->",
    highlight_arrow_linewidth: float = 1,
    highlight_text_offset_max: tuple[float, float] = (0.1, 0.2),
    highlight_text_offset_min: tuple[float, float] = (0.1, -0.2),
    highlight_style: Mapping[str, Mapping[str, Any]] = None,
    figsize: tuple[float, float] = (10, 5),
    hspace: float = None,
    vspace: float = None,
    save: bool = False,
    plot_name: str = "plot.pdf",
    level_markers: Union[dict, List[dict]] = None,
    show: bool = True,
) -> None:
    if subplot_groups is None:
        subplot_groups = [list(series.keys())]

    num_plots = len(subplot_groups)

    if subplot_layout:
        nrows, ncols = subplot_layout
        if nrows * ncols < num_plots:
            raise ValueError(f"O layout {subplot_layout} é pequeno demais para {num_plots} grupos.")
    else:
        nrows, ncols = num_plots, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    def get_setting(value, index):
        if isinstance(value, list):
            if value is x_ticks or value is y_ticks:
                if len(value) > 0 and isinstance(value[0], list):
                    return value[index] if index < len(value) else None
                return value
            return value[index] if index < len(value) else None
        return value

    for i, (ax, group) in enumerate(zip(axes, subplot_groups)):
        row = i // ncols
        col = i % ncols
        is_bottom_row = row == nrows - 1
        is_left_col = col == 0

        n = 0
        if group:
            n = max(len(series.get(name, [])) for name in group)

        for name in group:
            if name not in series:
                continue
            ys = series[name]
            xs = range(len(ys))

            raw_style = series_styles.get(name, {}) if series_styles else {}
            style = raw_style.copy()
            plot_label = style.pop("label", name)

            line, = ax.plot(xs, ys, label=plot_label, **style)

            mode = highlight.get(name) if highlight else None
            base_color = style.get("color", line.get_color())
            mcolor = highlight_color or base_color

            current_highlight_style = highlight_style.get(name, {}) if highlight_style else {}

            if mode in ("max", "both"):
                i_max = max(range(len(ys)), key=lambda j: ys[j])
                ax.plot(i_max, ys[i_max], marker=highlight_marker, markersize=highlight_markersize, color=mcolor)
                offset = current_highlight_style.get("highlight_offset_max", highlight_text_offset_max)
                text_position = (i_max + offset[0], ys[i_max] + offset[1])
                arrow_color = current_highlight_style.get("arrow_color", highlight_arrow_color or "dimgrey")
                arrow_style = current_highlight_style.get("arrow_style", highlight_arrow_style)
                arrow_width = current_highlight_style.get("arrow_linewidth", highlight_arrow_linewidth)
                text_color = current_highlight_style.get("text_color", highlight_text_color or "black")
                ax.annotate(
                    f"{ys[i_max]:.2f}",
                    xy=(i_max, ys[i_max]),
                    xytext=text_position,
                    arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, linewidth=arrow_width),
                    fontsize=highlight_text_size,
                    color=text_color,
                    va="bottom",
                    ha="center",
                )

            if mode in ("min", "both"):
                i_min = min(range(len(ys)), key=lambda j: ys[j])
                ax.plot(i_min, ys[i_min], marker=highlight_marker, markersize=highlight_markersize, color=mcolor)
                offset = current_highlight_style.get("highlight_offset_min", highlight_text_offset_min)
                text_position = (i_min + offset[0], ys[i_min] + offset[1])
                arrow_color = current_highlight_style.get("arrow_color", highlight_arrow_color or "dimgrey")
                arrow_style = current_highlight_style.get("arrow_style", highlight_arrow_style)
                arrow_width = current_highlight_style.get("arrow_linewidth", highlight_arrow_linewidth)
                text_color = current_highlight_style.get("text_color", highlight_text_color or "black")
                ax.annotate(
                    f"{ys[i_min]:.2f}",
                    xy=(i_min, ys[i_min]),
                    xytext=text_position,
                    arrowprops=dict(arrowstyle=arrow_style, color=arrow_color, linewidth=arrow_width),
                    fontsize=highlight_text_size,
                    color=text_color,
                    va="top",
                    ha="center",
                )

        if n > 0:
            current_num_yticks = get_setting(num_yticks, i)
            current_y_ticks = get_setting(y_ticks, i)

            current_num_xticks = get_setting(num_xticks, i)
            current_x_ticks = get_setting(x_ticks, i)

            current_first_step_xtick = get_setting(first_step_xtick, i)
            current_xtick_step = get_setting(xtick_step, i)

            if current_num_yticks or current_y_ticks:
                if current_num_yticks:
                    current_ylim = get_setting(ylim, i)
                    if current_ylim:
                        min_y, max_y = current_ylim
                    else:
                        min_y = float("inf")
                        max_y = float("-inf")
                        for name in group:
                            if name in series and len(series[name]) > 0:
                                min_y = min(min_y, min(series[name]))
                                max_y = max(max_y, max(series[name]))

                        if min_y == float("inf") or max_y == float("-inf"):
                            min_y, max_y = 0, 1.0

                        min_y = math.floor(min_y * 10) / 10
                        max_y = math.ceil(max_y * 10) / 10

                    yticks = np.linspace(min_y, max_y, current_num_yticks)
                    yticks = np.unique(yticks)
                else:
                    yticks = current_y_ticks
                ax.set_yticks(yticks)
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

            if current_x_ticks is not None:
                ax.set_xticks(current_x_ticks)
            elif current_num_xticks:
                xticks = np.linspace(1, n, current_num_xticks)
                ax.set_xticks(xticks.astype(int))
            elif current_first_step_xtick is not None:
                labels = [1]
                step = current_xtick_step if current_xtick_step is not None else 1
                next_label = 1 + current_first_step_xtick
                while next_label <= n:
                    labels.append(next_label)
                    next_label += step
                positions = [lbl - 1 for lbl in labels]
                labels = [lbl + xtick_offset for lbl in labels]
                ax.set_xticks(positions, labels)
            elif current_xtick_step is not None and current_xtick_step > 0:
                positions = list(range(0, n, current_xtick_step))
                labels = [pos + 1 + xtick_offset for pos in positions]
                ax.set_xticks(positions, labels)

        if hide_inner_ticks:
            if not is_bottom_row:
                ax.set_xticklabels([])
            if not is_left_col:
                ax.set_yticklabels([])

        if num_xticks and xtick_offset != 0 and n > 0 and x_ticks is None:
            fig.canvas.draw()
            current_ticks = ax.get_xticks()
            new_labels = [int(tick) + xtick_offset for tick in current_ticks]
            ax.set_xticklabels(new_labels)

        if tick_fontsize:
            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

        x_text = get_setting(xlabel, i)
        if isinstance(xlabel, list):
            ax.set_xlabel(x_text, fontsize=label_fontsize)
        elif is_bottom_row:
            ax.set_xlabel(x_text, fontsize=label_fontsize)

        y_text = get_setting(ylabel, i)
        if isinstance(ylabel, list):
            ax.set_ylabel(y_text, fontsize=label_fontsize)
        elif is_left_col:
            ax.set_ylabel(y_text, fontsize=label_fontsize)

        ax.set_title(get_setting(title, i), fontsize=title_fontsize)

        show_legend = False
        if legend_subplot_index == "all":
            show_legend = True
        elif isinstance(legend_subplot_index, list):
            if i in legend_subplot_index:
                show_legend = True
        elif i == legend_subplot_index:
            show_legend = True

        if show_legend:
            base_kwargs = {"loc": legend_loc, "fontsize": legend_fontsize}
            current_kwargs = get_setting(legend_kwargs, i)
            if current_kwargs:
                base_kwargs.update(current_kwargs)
            ax.legend(**base_kwargs)

        current_xlim = get_setting(xlim, i)
        if current_xlim:
            ax.set_xlim(*current_xlim)
        elif n > 0:
            if x_ticks and not isinstance(x_ticks, dict):
                ax.set_xlim(min(x_ticks), max(x_ticks))
            else:
                ax.set_xlim(0, n)

        current_ylim = get_setting(ylim, i)
        if current_ylim:
            ax.set_ylim(*current_ylim)

        if row_labels and col == ncols - 1:
            if row < len(row_labels) and row_labels[row]:
                ax.text(
                    1.05,
                    0.5,
                    row_labels[row],
                    transform=ax.transAxes,
                    rotation=270,
                    ha="left",
                    va="center",
                    fontsize=row_label_fontsize,
                    fontweight="bold",
                )

        middle_col_index = ncols // 2
        if row_suptitles and col == middle_col_index:
            if row < len(row_suptitles) and row_suptitles[row]:
                ax.text(
                    0.5,
                    1.2,
                    row_suptitles[row],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=row_suptitle_fontsize,
                    fontweight="bold",
                )

        current_markers = get_setting(level_markers, i)
        if current_markers:
            for label, x_pos in current_markers.items():
                ax.axvline(
                    x=x_pos,
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    zorder=0,
                )

                ax.text(
                    x=x_pos,
                    y=0.05,
                    s=label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="right",
                    va="bottom",
                    fontsize=18,
                    color="dimgrey",
                    fontweight="bold",
                )

    if figure_title:
        fig.suptitle(
            figure_title,
            fontsize=figure_title_fontsize,
            y=figure_title_y or 0.98,
        )

    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)

    if hspace is not None or vspace is not None or subplot_margins:
        margins = subplot_margins or {}
        plt.subplots_adjust(
            hspace=hspace or 0.3,
            wspace=vspace or 0.2,
            left=margins.get("left", 0.1),
            right=margins.get("right", 0.9),
            top=margins.get("top", 0.9),
            bottom=margins.get("bottom", 0.1),
        )
    else:
        fig.tight_layout()

    shift_amount = 0.04
    for i, ax in enumerate(axes):
        row = i // ncols

        if row != 0:
            if i in (9, 15):
                shift_amount += 0.04
            pos = ax.get_position()
            new_pos = [pos.x0, pos.y0 + shift_amount, pos.width, pos.height]
            ax.set_position(new_pos)

    if save:
        plt.savefig(plot_name)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_by_marker(ax, x_data, y_data, c_data, m_data, title):
    x_arr = np.array(x_data)
    y_arr = np.array(y_data)
    c_arr = np.array(c_data)
    m_arr = np.array(m_data)

    unique_markers = np.unique(m_arr)

    for marker in unique_markers:
        mask = m_arr == marker
        ax.scatter(
            x_arr[mask],
            y_arr[mask],
            c=c_arr[mask],
            marker=marker,
            s=340,
            alpha=0.9,
        )

    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Acurácia Máxima", fontsize=20)
    ax.set_ylabel("Tráfego (GB)", fontsize=20)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, linestyle="--", alpha=0.4)


def calculate_times_and_accs(exp_data, is_baseline):
    accs = exp_data.get("net_acc", [])
    epoch_times = exp_data.get("time_epoch_classifier", [])

    if is_baseline:
        times = np.cumsum(epoch_times).tolist()
        return times, accs

    transitions = exp_data.get("accuracy_transition", [])
    time_levels = exp_data.get("time_level", [])

    calculated_times = []
    current_cumulative_time = 0
    current_epoch_idx = 0

    for i, target_acc in enumerate(transitions):
        found_idx = -1
        for k in range(current_epoch_idx, len(accs)):
            if math.isclose(accs[k], target_acc, rel_tol=1e-9):
                found_idx = k
                break

        level_classifier_time = 0
        for k in range(current_epoch_idx, found_idx + 1):
            t = epoch_times[k]
            level_classifier_time += t
            current_cumulative_time += t
            calculated_times.append(current_cumulative_time)

        if i < len(time_levels) - 1:
            total_level_duration = time_levels[i]
            gan_time = total_level_duration - level_classifier_time
            gan_time = max(0, gan_time)
            current_cumulative_time += gan_time

        current_epoch_idx = found_idx + 1

    for k in range(current_epoch_idx, len(accs)):
        t = epoch_times[k]
        current_cumulative_time += t
        calculated_times.append(current_cumulative_time)

    return calculated_times, accs

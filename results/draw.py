import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def plot_algorithm_comparison(
    x_values,
    y_data,
    y_label="Execution time",
    line_width=1,
    grid_alpha=0.7,
    dpi=300,
    output_file="algorithm_comparison.png",
    visual_type="bar",  # bar, line
):
    """
    绘制算法比较图

    参数:
    x_values (list): X轴数据点
    y_data (dict): Y轴数据，格式为 {'图例名称': [y1, y2, ...]}
    y_label (str): Y轴标签 (默认: "depth")
    line_width (float): 线条宽度
    grid_alpha (float): 网格线透明度
    dpi (int): 输出图像分辨率
    output_file (str): 输出文件路径
    type (str): 图表类型，"bar"表示柱状图，"line"表示折线图
    """
    plt.figure(figsize=(4, 3))

    # 根据type参数决定绘图类型
    if visual_type == "bar":
        # 柱状图绘制
        x_positions = np.arange(len(x_values))
        bar_width = 0.35

        # 如果有多个数据系列，需要调整位置
        labels = list(y_data.keys())
        n_series = len(labels)

        # 定义新颜色
        colors = ["#FF9462", "#5B76FF"]

        # 先绘制柱状图，确保它们在网格之上
        if n_series > 1:
            # 多个数据系列的情况
            for i, (label, y_values) in enumerate(y_data.items()):
                color = colors[i % len(colors)]  # 循环使用颜色
                offset = (i - n_series / 2 + 0.5) * bar_width
                plt.bar(
                    x_positions + offset,
                    y_values,
                    bar_width,
                    color=color,
                    label=label,
                    linewidth=line_width,
                    edgecolor="black",  # 添加黑色边框
                    zorder=3,  # 设置柱子的层级高于网格
                )
        else:
            # 单个数据系列的情况
            for i, (label, y_values) in enumerate(y_data.items()):
                color = colors[i % len(colors)]  # 循环使用颜色
                plt.bar(
                    x_positions,
                    y_values,
                    bar_width,
                    color=color,
                    label=label,
                    linewidth=line_width,
                    edgecolor="black",  # 添加黑色边框
                    zorder=3,  # 设置柱子的层级高于网格
                )

        plt.xticks(x_positions, x_values, fontsize=12)

    else:  # 默认为折线图
        # 绘制每条数据线
        colors = ["#E06600", "#0000CC"]  # 加深的颜色：橙色和蓝色
        markers = ["o", "s"]  # 圆形和正方形标记
        for i, (label, y_values) in enumerate(y_data.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.plot(
                x_values,
                y_values,
                color=color,
                linewidth=line_width * 2,  # 加粗线条
                label=label,
                marker=marker,
                markersize=7 if marker == "s" else 8,  # 放大marker
                markeredgewidth=0.5,
                zorder=3,  # 设置线条的层级高于网格
            )
        plt.xticks(x_values, fontsize=12)

    # 添加网格并将其置于底层
    plt.grid(True, linestyle="--", alpha=grid_alpha, linewidth=0.8, zorder=0)

    # 设置图表元素
    plt.xlabel(
        "Qubit num" if visual_type == "bar" else "Measurement-reset time", fontsize=14, labelpad=10
    )
    plt.ylabel(y_label, fontsize=14, labelpad=10)
    plt.yticks(fontsize=12)

    # 设置Y轴范围
    all_ys = [val for vals in y_data.values() for val in vals]
    y_max = max(all_ys) * 1.2  # 自动计算并增加20%空间
    plt.ylim(0, y_max)

    plt.legend(
        frameon=True,
        framealpha=0.9,
        fontsize=12,
        handlelength=2,
        handletextpad=0.5,
    )

    plt.tight_layout()
    plt.savefig(output_file + ".pdf", bbox_inches="tight", dpi=dpi)
    plt.show()

import matplotlib.pyplot as plt
def draw_plot(xs, series_list, label_list, fname):
    fig, ax = plt.subplots()

    for i,series in enumerate(series_list):
        smask = np.isfinite(series)
        ax.plot(xs[smask], series[smask], linestyle='-', marker='o', label=label_list[i])

    legend = ax.legend(loc='best', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # plt.show()
    plt.savefig(fname)
    plt.close(fig)
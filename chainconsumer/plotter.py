import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm


class Plotter(object):

    def plot(self, parent, figsize="GROW", parameters=None, extents=None, filename=None,
             display=False, truth=None, legend=None):  # pragma: no cover
        if legend is None:
            legend = len(parent._chains) > 1

        # Get all parameters to plot, taking into account some of them
        # might be excluded colour parameters
        color_params = parent.config["color_params"]
        plot_color_params = parent.config["plot_color_params"]
        all_parameters = []
        for cp, ps, pc in zip(color_params, parent._parameters, plot_color_params):
            for p in ps:
                if (p != cp or pc) and p not in all_parameters:
                    all_parameters.append(p)

        # Calculate cmap extents
        unique_color_params = list(set(color_params))
        num_cax = len(unique_color_params)
        if None in unique_color_params:
            num_cax -= 1
        color_param_extents = {}
        for u in unique_color_params:
            umin, umax = np.inf, -np.inf
            for i, cp in enumerate(color_params):
                if cp is not None and u == cp:
                    data = parent._chains[i][:, parent._parameters[i].index(cp)]
                    umin = min(umin, data.min())
                    umax = max(umax, data.max())
            color_param_extents[u] = (umin, umax)

        if parameters is None:
            parameters = all_parameters
        elif isinstance(parameters, int):
            parameters = parent._all_parameters[:parameters]
        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()
        if truth is not None and isinstance(truth, list):
            truth = truth[:len(parameters)]
        if isinstance(figsize, float):
            grow_size = figsize
            figsize = "GROW"
        else:
            grow_size = 1.0

        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5 + (1 if num_cax > 0 else 0), 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            elif figsize.upper() == "GROW":
                figsize = (grow_size * len(parameters) + num_cax * 1.0, grow_size * len(parameters))

            else:
                raise ValueError("Unknown figure size %s" % figsize)
        elif isinstance(figsize, float):
            figsize = (figsize * grow_size * len(parameters), figsize * grow_size * len(parameters))

        assert truth is None or isinstance(truth, dict) or \
               (isinstance(truth, list) and len(truth) == len(parameters)), \
            "Have a list of %d parameters and %d truth values" % (len(parameters), len(truth))

        assert extents is None or isinstance(extents, dict) or \
               (isinstance(extents, list) and len(extents) == len(parameters)), \
            "Have a list of %d parameters and %d extent values" % (len(parameters), len(extents))

        if truth is not None and isinstance(truth, list):
            truth = dict((p, t) for p, t in zip(parameters, truth))

        if extents is not None and isinstance(extents, list):
            extents = dict((p, e) for p, e in zip(parameters, extents))

        plot_hists = parent.config["plot_hists"]
        flip = (len(parameters) == 2 and plot_hists and parent.config["flip"])

        fig, axes, params1, params2, extents = self._get_figure(parent, parameters, figsize=figsize,
                                                                flip=flip, external_extents=extents)
        axl = axes.ravel().tolist()
        summary = parent.config["summary"]
        fit_values = parent.get_summary(squeeze=False)

        if summary is None:
            summary = len(parameters) < 5 and len(parent._chains) == 1
        if len(parent._chains) == 1:
            parent._logger.debug("Plotting surfaces for chain of dimension %s" %
                               (parent._chains[0].shape,))
        else:
            parent._logger.debug("Plotting surfaces for %d chains" % len(parent._chains))
        cbar_done = []
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j:
                    continue
                ax = axes[i, j]
                do_flip = (flip and i == len(params1) - 1)
                if plot_hists and i == j:
                    max_val = None
                    for ii, (chain, weights, parameters, fit, grid) in \
                            enumerate(zip(parent._chains, parent._weights, parent._parameters, fit_values, parent._grids)):
                        if p1 not in parameters:
                            continue
                        index = parameters.index(p1)

                        m = self._plot_bars(parent, ii, ax, p1, chain[:, index], weights, grid=grid, fit_values=fit[p1],
                                            flip=do_flip,
                                            summary=summary, truth=truth, extents=extents[p1])
                        if max_val is None or m > max_val:
                            max_val = m
                    if do_flip:
                        ax.set_xlim(0, 1.1 * max_val)
                    else:
                        ax.set_ylim(0, 1.1 * max_val)

                else:
                    for ii, (chain, parameters, fit, weights, grid) in \
                            enumerate(zip(parent._chains, parent._parameters, fit_values, parent._weights, parent._grids)):
                        if p1 not in parameters or p2 not in parameters:
                            continue
                        i1 = parameters.index(p1)
                        i2 = parameters.index(p2)
                        color_data = None
                        extent = None
                        if color_params[ii] is not None:
                            color_index = parameters.index(color_params[ii])
                            color_data = chain[:, color_index]
                            extent = color_param_extents[color_params[ii]]
                        h = self._plot_contour(parent, ii, ax, chain[:, i2], chain[:, i1], weights, p1, p2,
                                               grid, truth=truth, color_data=color_data, color_extent=extent)
                        if h is not None and color_params[ii] not in cbar_done:
                            cbar_done.append(color_params[ii])
                            aspect = figsize[1] / 0.15
                            fraction = 0.85 / figsize[0]
                            cbar = fig.colorbar(h, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                            cbar.set_label(color_params[ii], fontsize=14)
                            cbar.solids.set(alpha=1)

        colors = parent.config["colors"]
        linestyles = parent.config["linestyles"]
        linewidths = parent.config["linewidths"]
        if parent._names is not None and legend:
            ax = axes[0, -1]
            artists = [plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw)
                       for n, c, ls, lw in zip(parent._names, colors, linestyles, linewidths) if n is not None]
            location = "center" if len(parameters) > 1 else 1
            ax.legend(artists, parent._names, loc=location, frameon=False)
        fig.canvas.draw()
        for ax in axes[-1, :]:
            offset = ax.get_xaxis().get_offset_text()
            ax.set_xlabel('{0} {1}'.format(ax.get_xlabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
            offset.set_visible(False)
        for ax in axes[:, 0]:
            offset = ax.get_yaxis().get_offset_text()
            ax.set_ylabel('{0} {1}'.format(ax.get_ylabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
            offset.set_visible(False)
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()

        return fig

    def plot_walks(self, parent, parameters=None, truth=None, extents=None, display=False,
                   filename=None, chains=None, convolve=None, figsize=None,
                   plot_weights=True, plot_posterior=True, log_weight=None): # pragma: no cover

        if truth is not None and isinstance(truth, np.ndarray):
            truth = truth.tolist()

        if chains is None:
            chains = list(range(len(parent._chains)))
        else:
            if isinstance(chains, str) or isinstance(chains, int):
                chains = [chains]
            chains = [parent._get_chain(c) for c in chains]

        all_parameters2 = [p for i in chains for p in parent._parameters[i]]
        all_parameters = []
        for p in all_parameters2:
            if p not in all_parameters:
                all_parameters.append(p)

        if parameters is None:
            parameters = all_parameters
        elif isinstance(parameters, int):
            parameters = parent.all_parameters[:parameters]

        if truth is not None and isinstance(truth, list):
            truth = truth[:len(parameters)]

        assert truth is None or isinstance(truth, dict) or \
               (isinstance(truth, list) and len(truth) == len(parameters)), \
            "Have a list of %d parameters and %d truth values" % (len(parameters), len(truth))

        assert extents is None or isinstance(extents, dict) or \
               (isinstance(extents, list) and len(extents) == len(parameters)), \
            "Have a list of %d parameters and %d extent values" % (len(parameters), len(extents))

        if truth is not None and isinstance(truth, list):
            truth = dict((p, t) for p, t in zip(parameters, truth))
        if truth is None:
            truth = {}

        if extents is not None and isinstance(extents, list):
            extents = dict((p, e) for p, e in zip(parameters, extents))
        if extents is None:
            extents = {}

        n = len(parameters)
        extra = 0
        if plot_weights:
            plot_weights = plot_weights and np.any([np.any(parent._weights[c] != 1.0) for c in chains])

        plot_posterior = plot_posterior and np.any([parent._posteriors[c] is not None for c in chains])

        if plot_weights:
            extra += 1
        if plot_posterior:
            extra += 1

        if figsize is None:
            figsize = (8, 0.75 + (n + extra))

        colors = parent.config["colors"]

        if parent.config["usetex"]:
            plt.rc('text', usetex=True)
        if parent.config["serif"]:
            plt.rc('font', family='serif')
        else:
            plt.rc('font', family='sans-serif')

        fig, axes = plt.subplots(figsize=figsize, nrows=n + extra, squeeze=False, sharex=True)

        for i, axes_row in enumerate(axes):
            ax = axes_row[0]
            if i >= extra:
                p = parameters[i - n]
                for index in chains:
                    if p in parent._parameters[index]:
                        chain_row = parent._chains[index][:, parent._parameters[index].index(p)]
                        self._plot_walk(parent, ax, p, chain_row, truth=truth.get(p),
                                        extents=extents.get(p), convolve=convolve, color=colors[index])
                        truth[p] = None
            else:
                if i == 0 and plot_posterior:
                    for index in chains:
                        if parent._posteriors[index] is not None:
                            self._plot_walk(parent, ax, "$\log(P)$", parent._posteriors[index] - parent._posteriors[index].max(),
                                            convolve=convolve, color=colors[index])
                else:
                    if log_weight is None:
                        log_weight = np.any([self._weights[index].mean() < 0.1 for index in chains])
                    if log_weight:
                        for index in chains:
                            self._plot_walk(parent, ax, r"$\log_{10}(w)$", np.log10(parent._weights[index]),
                                            convolve=convolve, color=colors[index])
                    else:
                        for index in chains:
                            self._plot_walk(parent, ax, "$w$", parent._weights[index],
                                            convolve=convolve, color=colors[index])

        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.05)
        if display:
            plt.show()
        return fig

    def _get_figure(self, parent, all_parameters, flip, figsize=(5, 5), external_extents=None):  # pragma: no cover
        n = len(all_parameters)
        max_ticks = parent.config["max_ticks"]
        plot_hists = parent.config["plot_hists"]
        label_font_size = parent.config["label_font_size"]
        tick_font_size = parent.config["tick_font_size"]
        diagonal_tick_labels = parent.config["diagonal_tick_labels"]

        if not plot_hists:
            n -= 1

        if n == 2 and plot_hists and flip:
            gridspec_kw = {'width_ratios': [3, 1], 'height_ratios': [1, 3]}
        else:
            gridspec_kw = {}

        if parent.config["usetex"]:
            plt.rc('text', usetex=True)
        if parent.config["serif"]:
            plt.rc('font', family='serif')
        else:
            plt.rc('font', family='sans-serif')

        fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=0.05)

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-3, 4))

        extents = {}
        for p in all_parameters:
            min_val = None
            max_val = None
            if external_extents is not None and p in external_extents:
                min_val, max_val = external_extents[p]
            else:
                for i, (chain, parameters, w) in enumerate(zip(parent._chains, parent._parameters, parent._weights)):
                    if p not in parameters:
                        continue
                    index = parameters.index(p)
                    if parent._grids[i]:
                        min_prop = chain[:, index].min()
                        max_prop = chain[:, index].max()
                    else:
                        min_prop, max_prop = parent._get_extent(chain[:, index], w)
                    if min_val is None or min_prop < min_val:
                        min_val = min_prop
                    if max_val is None or max_prop > max_val:
                        max_val = max_prop
            extents[p] = (min_val, max_val)

        if plot_hists:
            params1 = all_parameters
            params2 = all_parameters
        else:
            params1 = all_parameters[1:]
            params2 = all_parameters[:-1]
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                ax = axes[i, j]
                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    if i != n - 1 or (flip and j == n - 1):
                        ax.set_xticks([])
                    else:
                        display_x_ticks = True
                        if isinstance(p2, str):
                            ax.set_xlabel(p2, fontsize=label_font_size)
                    if j != 0 or (plot_hists and i == 0):
                        ax.set_yticks([])
                    else:
                        display_y_ticks = True
                        if isinstance(p1, str):
                            ax.set_ylabel(p1, fontsize=label_font_size)
                    if display_x_ticks:
                        if diagonal_tick_labels:
                            [l.set_rotation(45) for l in ax.get_xticklabels()]
                        [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]
                        ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                        ax.xaxis.set_major_formatter(formatter)
                    if display_y_ticks:
                        if diagonal_tick_labels:
                            [l.set_rotation(45) for l in ax.get_yticklabels()]
                        [l.set_fontsize(tick_font_size) for l in ax.get_yticklabels()]
                        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                        ax.yaxis.set_major_formatter(formatter)
                    if i != j or not plot_hists:
                        ax.set_ylim(extents[p1])
                    elif flip and i == 1:
                        ax.set_ylim(extents[p1])
                    ax.set_xlim(extents[p2])

        return fig, axes, params1, params2, extents

    def _plot_contour(self, parent, iindex, ax, x, y, w, px, py, grid, truth=None, color_data=None, color_extent=None):  # pragma: no cover

        levels = 1.0 - np.exp(-0.5 * parent.config["sigmas"] ** 2)
        h = None
        cloud = parent.config["cloud"][iindex]
        smooth = parent.config["smooth"][iindex]
        colour = parent.config["colors"][iindex]
        bins = parent.config["bins"][iindex]
        shade = parent.config["shade"][iindex]
        shade_alpha = parent.config["shade_alpha"][iindex]
        linestyle = parent.config["linestyles"][iindex]
        linewidth = parent.config["linewidths"][iindex]
        cmap = parent.config["cmaps"][iindex]

        if grid:
            binsx = parent._get_grid_bins(x)
            binsy = parent._get_grid_bins(y)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=[binsx, binsy], weights=w)
        else:
            bins, smooth = parent._get_smoothed_bins(smooth, bins, marginalsied=False)
            hist, x_bins, y_bins = np.histogram2d(x, y, bins=bins, weights=w)

        colours = parent._scale_colours(colour, len(levels))
        colours2 = [parent._scale_colour(colours[0], 0.7)] + \
                   [parent._scale_colour(c, 0.8) for c in colours[:-1]]

        x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        if smooth:
            hist = gaussian_filter(hist, smooth, mode=parent._gauss_mode)
        hist[hist == 0] = 1E-16
        vals = parent._convert_to_stdev(hist.T)
        if cloud:
            n = parent.config["num_cloud"][iindex]
            skip = max(1, int(x.size / n))
            kwargs = {"c": colours[1], "alpha": 0.3}
            if color_data is not None:
                kwargs["c"] = color_data[::skip]
                kwargs["cmap"] = cmap
                if color_extent is not None:
                    kwargs["vmin"] = color_extent[0]
                    kwargs["vmax"] = color_extent[1]

            h = ax.scatter(x[::skip], y[::skip], s=10, marker=".", edgecolors="none", **kwargs)
            if color_data is None:
                h = None

        if shade:
            ax.contourf(x_centers, y_centers, vals, levels=levels, colors=colours,
                        alpha=shade_alpha)
        ax.contour(x_centers, y_centers, vals, levels=levels, colors=colours2,
                   linestyles=linestyle, linewidths=linewidth)

        if truth is not None:
            truth_value = truth.get(px)
            if truth_value is not None:
                ax.axhline(truth_value, **parent.config_truth)
            truth_value = truth.get(py)
            if truth_value is not None:
                ax.axvline(truth_value, **parent.config_truth)
        return h

    def _plot_bars(self, parent, iindex, ax, parameter, chain_row, weights, flip=False, summary=False, fit_values=None,
                   truth=None, extents=None, grid=False):  # pragma: no cover

        # Get values from config
        kde = parent.config["kde"][iindex]
        colour = parent.config["colors"][iindex]
        linestyle = parent.config["linestyles"][iindex]
        bar_shade = parent.config["bar_shade"][iindex]
        linewidth = parent.config["linewidths"][iindex]
        bins = parent.config["bins"][iindex]
        smooth = parent.config["smooth"][iindex]
        title_size = parent.config["label_font_size"]

        bins, smooth = parent._get_smoothed_bins(smooth, bins)
        if grid:
            bins = parent._get_grid_bins(chain_row)
        else:
            bins = np.linspace(extents[0], extents[1], bins)
        hist, edges = np.histogram(chain_row, bins=bins, normed=True, weights=weights)
        edge_center = 0.5 * (edges[:-1] + edges[1:])
        if smooth:
            hist = gaussian_filter(hist, smooth, mode=parent._gauss_mode)
        if kde:
            assert np.all(weights == 1.0), "You can only use KDE if your weights are all one. " \
                                           "If you would like weights, please vote for this issue: " \
                                           "https://github.com/scikit-learn/scikit-learn/issues/4394"
            pdf = sm.nonparametric.KDEUnivariate(chain_row)
            pdf.fit()
            xs = np.linspace(extents[0], extents[1], 100)
            if flip:
                ax.plot(pdf.evaluate(xs), xs, color=colour, ls=linestyle, lw=linewidth)
            else:
                ax.plot(xs, pdf.evaluate(xs), color=colour, ls=linestyle, lw=linewidth)
            interpolator = pdf.evaluate
        else:
            if smooth:
                if flip:
                    ax.plot(hist, edge_center, color=colour, ls=linestyle, lw=linewidth)
                else:
                    ax.plot(edge_center, hist, color=colour, ls=linestyle, lw=linewidth)
            else:
                if flip:
                    orientation = "horizontal"
                else:
                    orientation = "vertical"
                ax.hist(edge_center, weights=hist, bins=edges, histtype="step",
                        color=colour, orientation=orientation, ls=linestyle, lw=linewidth)
            interp_type = "linear" if smooth else "nearest"
            interpolator = interp1d(edge_center, hist, kind=interp_type)

        if bar_shade and fit_values is not None:
            lower = fit_values[0]
            upper = fit_values[2]
            if lower is not None and upper is not None:
                if lower < edge_center.min():
                    lower = edge_center.min()
                if upper > edge_center.max():
                    upper = edge_center.max()
                x = np.linspace(lower, upper, 1000)
                if flip:
                    ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x),
                                     color=colour, alpha=0.2)
                else:
                    ax.fill_between(x, np.zeros(x.shape), interpolator(x),
                                    color=colour, alpha=0.2)
                if summary:
                    if isinstance(parameter, str):
                        ax.set_title(r"$%s = %s$" % (parameter.strip("$"),
                                                     parent.get_parameter_text(*fit_values)), fontsize=title_size)
                    else:
                        ax.set_title(r"$%s$" % (parent.get_parameter_text(*fit_values)), fontsize=title_size)
        if truth is not None:
            truth_value = truth.get(parameter)
            if truth_value is not None:
                if flip:
                    ax.axhline(truth_value, **parent.config_truth)
                else:
                    ax.axvline(truth_value, **parent.config_truth)
        return hist.max()

    def _plot_walk(self, parent, ax, parameter, data, truth=None, extents=None,
                   convolve=None, color=None):  # pragma: no cover
        if extents is not None:
            ax.set_ylim(extents)
        assert convolve is None or isinstance(convolve, int), \
            "Convolve must be an integer pixel window width"
        x = np.arange(data.size)
        ax.set_xlim(0, x[-1])
        ax.set_ylabel(parameter)
        if color is None:
            color = "#0345A1"
        ax.scatter(x, data, c=color, s=2, marker=".", edgecolors="none", alpha=0.5)
        max_ticks = parent.config["max_ticks"]
        ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))

        if convolve is not None:
            color2 = parent._scale_colour(color, 0.5)
            filt = np.ones(convolve) / convolve
            filtered = np.convolve(data, filt, mode="same")
            ax.plot(x[:-1], filtered[:-1], ls=':', color=color2, alpha=1)
        if truth is not None:
            ax.axhline(truth, **parent.config_truth)

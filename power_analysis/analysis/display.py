import functools

import matplotlib.pylab as plt
import numpy as np
from IPython.core.display import display
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
from tqdm.auto import tqdm

from .arch import Instructions
from .correlation import BaseCorrelation
from ..utils import MixinLogger


def display_df(df: pd.DataFrame):
    """Visualize the DataFrame coloring with a gradient based on the correlation value."""
    # generate the r-value only dataframe
    # to use as colormap for the gradient
    rdf = df.copy()

    for column in rdf:
        new_column = []

        for b, r in rdf[column]:
            new_column.append(r)

        rdf[column] = pd.Series(new_column)

    pd.set_option('display.precision', 2)  # FIXME: doesn't work
    # pd.set_option('display.max_columns', None)
    # pd.options.display.float_format = '{:.2f}'.format
    cm = plt.get_cmap('Greens')
    df_elaborated = df.head().style \
        .background_gradient(cm, axis=None, vmin=.5, gmap=rdf) \
        .format(formatter='{0[0]}:{0[1]:0.2f}')
    display(df_elaborated)

    return df_elaborated.data


class Plot(MixinLogger):
    """Utility class to plot stuff"""

    def __init__(self, correlation: BaseCorrelation):
        """Initialize the instance with the data from the correlation interface
        of the experiment."""
        super(Plot, self).__init__()

        self._correlation = correlation

    @property
    def samples_for_instruction(self):
        return self._correlation.capture.samples_for_instruction

    def statistics(self):
        plt.figure(figsize=(10, 5))

        avgs, stds = self._correlation.stats_from_traces()

        x = range(len(avgs))
        y = avgs
        yerr = stds

        plt.ylim([min(y) - max(yerr), max(y) + max(yerr)])

        plt.errorbar(x, y, yerr=yerr, ecolor='grey')

        plt.show()

    def _annotate_w_instructions(self, instructions, start=None, colormap_name='tab20b'):
        if self.samples_for_instruction is None:
            self.logger.warning("attribute 'samples_for_instruction' not set so we cannot annotate instructions")
            return

        offset = start or 0

        n = len(instructions)
        cm = plt.get_cmap(colormap_name)
        colors = cm.colors
        # we want that the instructions' mnemonic visualized
        # vertically with the x coord in the middle of the execution span
        # and the y-coord a little separated by the bottom of the figure
        #  https://stackoverflow.com/questions/41705024/matplotlib-override-y-position-with-pixel-position
        import matplotlib.transforms as transforms
        ax = plt.gca()
        transformation = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # the first instruction should have only one cycle
        is_first = True

        for index, instruction in enumerate(instructions):
            begin = offset

            if is_first:
                cycles = 1
                is_first = False
            else:
                cycles = instruction.cycles

            end = offset + cycles * self.samples_for_instruction
            plt.axvspan(begin, end, alpha=.5, color=colors[index % cm.N])

            plt.text(
                begin + cycles * self.samples_for_instruction / 2,
                .05,  # this could be improved (width/20 as a distance from the border)
                instruction.mnemonic,
                rotation=90,
                transform=transformation
            )

            offset = end

    def classification(self, limits=None, colormap_name='tab20b', instructions: Instructions = None, **kwargs):
        plt.figure(figsize=(10, 5))

        self._set_ticks()
        self._extra()

        for weight, stats in zip(self._correlation.classes, self._correlation.statistics):
            plt.plot(stats[0], label=str(weight), **kwargs)

        # plt.grid()
        # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        if limits is not None:
            xlim, ylim = limits
            plt.xlim(xlim)
            plt.ylim(ylim)

        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #        mode="expand", borderaxespad=0, ncol=3)
        plt.tight_layout()

        if instructions:
            self._annotate_w_instructions(instructions, start=instructions.start, colormap_name=colormap_name)

        plt.show()

    def correlations(self, title, start, count=10):
        """Plot multiple diagrams with represented the relation (hamming weight, power consumption)
        for each clock cycle in a time window."""
        fig, axs = plt.subplots(nrows=count, ncols=4, sharex='all', figsize=(3 * 4, 5 * count))

        plt.subplots_adjust(wspace=.5, hspace=.1)

        offset = 0

        n_weights = len(self._correlation.statistics)

        for row in axs:  # FIXME: if nrows is 1 then we don't have a matrix
            for col in row:
                col.set_title(str(start + offset))
                col.grid(linestyle=':')
                # col.set_ylim([-0.150, 0.150])
                col.errorbar(
                    list(range(0, n_weights)),
                    self._correlation.statistics[:, 0, start + offset],
                    yerr=self._correlation.statistics[:, 1, start + offset], fmt='.')
                col.set_xticks(list(range(n_weights)))
                offset += 1

        position_middle_title = (fig.subplotpars.right + fig.subplotpars.left) / 2
        fig.suptitle(title, ha='left', x=position_middle_title)

        fig.show()

    def pearson(self, *args, r=None, p=0.05, limits=(None, [-1, 1]), instructions=None, colormap_name='tab20b', **kwargs):
        """Use the parameters `r` and `p` to filter and show only the relevant correlations
        otherwise both correlations and p-values will be shown."""
        pearsons_and_pvalues = self._correlation.pearson(*args, **kwargs)

        pearsons = pearsons_and_pvalues[:, 0]
        pvalues = pearsons_and_pvalues[:, 1]

        plt.figure()

        self._set_ticks()

        self._extra()

        if r:
            pearsons = [pr if abs(pr) > r else 0 for pr in pearsons]

        if p:
            pearsons = [pr if pv < p else 0 for pr, pv in zip(pearsons, pvalues)]

        if r or p:
            plt.plot(pearsons, drawstyle='steps')  # , markerfmt='')
        else:
            plt.plot(pearsons)
            plt.plot(pvalues)

        if instructions:
            self._annotate_w_instructions(instructions, start=instructions.start, colormap_name=colormap_name)

        if limits is not None:
            xlim, ylim = limits
            plt.xlim(xlim)
            plt.ylim(ylim)

        plt.show()

    # FIXME: usually the Correlation doesn't have `pearsons` attribute
    def dataframe_from_pearsons(self, instructions=None, threshold=None):
        df = self._correlation.pearsons_df(instructions=instructions, threshold=threshold)
        return display_df(df)

    def _set_ticks(self):
        if not self.samples_for_instruction:
            self.logger.warning("no samples_for_instruction set, so no ticks will be displayed")
            return

        ax = plt.axes()

        ax.xaxis.set_minor_locator(plt.MultipleLocator(self.samples_for_instruction))

        plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    def _extra(self):
        """Subclasses need to implement this if they want something extra for the plot."""
        pass

    @classmethod
    def plot(cls, data, start: int = 0, end=None, figure=True, limits=None):
        """Generic plot with ticks setup to indicate instructions timing."""
        if figure:
            plt.figure()

        if limits is not None:
            xlim, ylim = limits
            plt.xlim(xlim)
            plt.ylim(ylim)

        range_ = slice(start, end or len(data))

        plt.plot(data[range_])

        plt.show()

    @classmethod
    @functools.lru_cache()
    def matrix(cls, length, capture, corr_class, progress_bar=0, only_diagonal=False, **kwargs):
        corr = corr_class.matrix(capture, length=length, only_diagonal=only_diagonal, **kwargs)

        # the idea here is that the progress_bar argument can indicate
        # which level of the progress bar you want activated
        progress_bar_pearson = progress_bar >> 1

        pearsons = [(l, r, c, c.pearson(progress_bar=progress_bar_pearson))
                    for l, r, c in tqdm(corr, disable=progress_bar == 0)]

        fig, axs = plt.subplots(
            nrows=length,
            ncols=length if not only_diagonal else 1,
            sharex='all',
            sharey='all',
            constrained_layout=True,
        )

        for left, right, corr, pearson_and_pvalue in tqdm(pearsons, disable=progress_bar == 0):
            ax = axs[left, right] if not only_diagonal else axs[left]

            ax.set_ylim([-1, 1])

            # we extract the r value
            pearson = pearson_and_pvalue[:, 0]

            pearson_sorted_arg = np.argsort(pearson)

            pearson_max_index = pearson_sorted_arg[-1]
            pearson_max = pearson[pearson_max_index]

            ax.plot(pearson)

            ax.set_title(f"{pearson_max:.2f} @ {pearson_max_index}")

        # fig.tight_layout()
        return np.array(pearsons, dtype=object)

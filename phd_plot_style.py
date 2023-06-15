import math
import copy
import matplotlib as mpl

class phd_revtex_plots:

    def __init__(self, revtex_fontsize=10, scale_factor=1.0):
        self.gold_ratio = (1. + math.sqrt(5)) / 2.
        self.tex_pt_to_inch = 1. / 72.27
        self.revtex_sizes_active = self._get_revtex_sizes(revtex_fontsize)
        self.revtex_rc_active = self._get_revtex_rc(scale_factor)

#         # PRL DEFAULT SIZE given on webpage (https://journals.aps.org/prl/info/infoL.html)
#         self.prl_colum_width_in_round  = 3.375 
        
    def _get_revtex_sizes(self, fontsize):
        rextex_fonts = {}
        if fontsize == 10:
            rextex_font_size = {
                "text.normalsize": 10,
                "text.small": 9,
                "text.footnotesize": 8,
                "text.scriptsize": 7,
                "text.tiny": 5,
                "text.large": 12,
                "page.textwidth": 510,
                "page.columnsep": 18,
                "page.columnwidth": 246, # = (textwidth - columnsep) / 2
            }
        elif fontsize == 11:
            rextex_font_size = {
                "text.normalsize": 11,
                "text.small": 10,
                "text.footnotesize": 9,
                "text.scriptsize": 8,
                "text.tiny": 6,
                "text.large": 12,
                "page.textwidth": 468,
                "page.columnsep": 10,
                "page.columnwidth": 229, # = (textwidth - columnsep) / 2
            }
        elif rextex_font_size == 12:
            rextex_fonts = {
                "text.normalsize": 12,
                "text.small": 11,
                "text.footnotesize": 10,
                "text.scriptsize": 8,
                "text.tiny": 6,
                "text.large": 14,
                "page.textwidth": 468,
                "page.columnsep": 10,
                "page.columnwidth": 229, # = (textwidth - columnsep) / 2
            }
        else:
            raise IOError("Fontsize Needs to be one of 10, 11, 12 (pt)")
        return rextex_font_size

    def _get_revtex_rc(self, scale_factor=1.0):    
        
        fig_width = self.revtex_sizes_active["page.columnwidth"] * self.tex_pt_to_inch
        fig_height = self.revtex_sizes_active["page.columnwidth"] * self.tex_pt_to_inch / self.gold_ratio

        
        manually_scaleable_options = {
            "figure.titlesize": self.revtex_sizes_active["text.normalsize"],
            # "figure.figsize" = (fig_width, fig_height) ## can"t scale here

            "font.size": self.revtex_sizes_active["text.normalsize"],
            "axes.labelsize": self.revtex_sizes_active["text.small"],
            "axes.titlesize": self.revtex_sizes_active["text.normalsize"],
            "xtick.labelsize": self.revtex_sizes_active["text.small"],
            "ytick.labelsize": self.revtex_sizes_active["text.small"],
            "legend.fontsize": self.revtex_sizes_active["text.footnotesize"],

            "axes.labelpad": 0.2 * self.revtex_sizes_active["text.small"],
            "axes.titlepad": 0.3 * self.revtex_sizes_active["text.normalsize"],
            
            "axes.linewidth": 1.0,
            "grid.linewidth": 1.0,
            "lines.linewidth": 0.75, #1.0
            "lines.markersize": 1.5,
            "patch.linewidth": 0.5, #1.0

            "errorbar.capsize": 0.75,
            
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,

            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            
            "xtick.major.pad": 0.20 * self.revtex_sizes_active["text.normalsize"],
            "ytick.major.pad": 0.20 * self.revtex_sizes_active["text.normalsize"],
            "xtick.minor.pad": 0.20 * self.revtex_sizes_active["text.normalsize"],
            "ytick.minor.pad": 0.20 * self.revtex_sizes_active["text.normalsize"],
            
            }
        
        manually_scaleable_options.update((x, y * scale_factor) for x, y in manually_scaleable_options.items())
        
        manually_scaleable_options["figure.figsize"] = (scale_factor*fig_width, scale_factor*fig_height)

        non_manually_scaleable_options = {
            # animation options not included
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "text.hinting": "auto",
            "pgf.rcfonts": False,
            "pgf.texsystem": "pdflatex",
            "pgf.preamble": "\n".join([
                 r"\usepackage{amsmath}",
            ]),

            "axes.xmargin": 0.05,  # scale axes range to include space outside data
            "axes.ymargin": 0.05,

            "figure.dpi": 100,
            "figure.autolayout": False,
            "figure.titleweight": "normal",
            "figure.frameon": False,
            "figure.facecolor": "white",

            # Font Family
            "font.family": "serif",
            "font.style": "normal",
            "font.variant": "normal",
            "font.weight": "medium",
            "font.stretch": "normal",
            "font.serif": "Computer Modern Roman, Times, Palatino, New Century Schoolbook, Bookman",

            # Legend as fraction of fontsize
            "legend.loc": "best",
            "legend.frameon": False,
            "legend.facecolor": "inherit",
            "legend.numpoints": 1,
            "legend.fancybox": False,
            "legend.labelspacing": 0.4,
            "legend.handlelength": 1.0,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.5,  # Fraction of Fontsize
            "legend.borderpad": 0.25,
            "legend.columnspacing": 0.5,
            "legend.handleheight": 0.5,
            "legend.markerscale": 2.0,
            "legend.framealpha": 0.8,
            "legend.scatterpoints": 1,
            "legend.shadow": False,

            # Axes
            "axes.autolimit_mode": "data",
            "axes.axisbelow": "line",
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "axes.formatter.limits": [-7, 7],
            "axes.formatter.min_exponent": 0,
            "axes.formatter.offset_threshold": 4,
            "axes.formatter.use_locale": False,
            "axes.formatter.use_mathtext": False,
            "axes.formatter.useoffset": True,
            "axes.grid": False,
            "axes.grid.axis": "both",
            "axes.grid.which": "major",
            "axes.labelcolor": "black",
            "axes.labelweight": "normal",
            # "axes.linewidth": 0.8,
            #             "axes.prop_cycle": cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]),
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": True,
            "axes.spines.top": True,
            #            "axes.titlepad": 6.0,
            #            "axes.titlesize": "large",
            "axes.titleweight": "normal",
            "axes.unicode_minus": False,

            # Savefig
            "savefig.bbox": None,
            "savefig.directory": "~",
            "savefig.dpi": 600,
            "savefig.edgecolor": "white",
            "savefig.facecolor": "none",  # makes background transparent
            "savefig.format": "pdf",
            "savefig.orientation": "portrait",
            "savefig.pad_inches": 0.0,
            "savefig.transparent": False,

            # Tick Parameters
            "xtick.color": "k",
            "xtick.direction": "out",
            "xtick.minor.visible": False,
            "xtick.bottom": True,
            "xtick.major.bottom": True,
            "xtick.minor.bottom": False,
            "xtick.top": False,
            "xtick.major.top": False,
            "xtick.minor.top": True,
            #
            "ytick.color": "k",
            "ytick.direction": "out",
            "ytick.minor.visible": False,
            "ytick.left": True,
            "ytick.major.left": True,
            "ytick.minor.left": False,
            "ytick.right": False,
            "ytick.major.right": False,
            "ytick.minor.right": False,

            "image.cmap": "inferno"
        }

        my_context_dict = copy.deepcopy(mpl.rcParamsDefault)
        my_context_dict.update(manually_scaleable_options)
        my_context_dict.update(non_manually_scaleable_options)

        return my_context_dict
        

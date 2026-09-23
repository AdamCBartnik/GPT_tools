
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import PathCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

class SnappingCursor:
    """
    A cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.

    line_list may hold Line2D objects (from ax.plot) or PathCollections
    (from ax.scatter); scatter points are highlighted in their own colormap color.
    """
    def __init__(self, fig, ax, line_list):
        self.ax = ax
        self.fig = fig
        self.canvas = self.fig.canvas
        self.x = []
        self.y = []
        self.point_colors = []
        for l in line_list:
            if isinstance(l, PathCollection):
                xy = np.asarray(l.get_offsets())
                self.x.append(xy[:, 0])
                self.y.append(xy[:, 1])
                self.point_colors.append(l.to_rgba(l.get_array()))
            else:
                self.x.append(np.asarray(l.get_data()[0]))
                self.y.append(np.asarray(l.get_data()[1]))
                self.point_colors.append(np.tile(mcolors.to_rgba(l.get_color()), (len(self.x[-1]), 1)))
        self._last_index = None
        self.text = ax.text(0.5, 0.5, '', color='white', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment='left')
        self.circ = plt.Rectangle((np.nan,np.nan), 0.1, 0.1, facecolor='white', edgecolor='r', alpha = 0.5)
        self.ax.add_patch(self.circ)
        self.screen_size = ax.get_figure().get_size_inches()
        self.pos = [0, 0]
        self.data_index = 0
        self.which_line = 0
        
    def set_visible(self, visible):
        self.text.set_visible(visible)
        self.circ.set_visible(visible)

    def on_mouse_move(self, event):       
        if (event.inaxes != self.ax) or (event.button == MouseButton.RIGHT):
            self._last_index = None
            self.set_visible(False)
        else:
            self.set_visible(True)
            x, y = event.xdata, event.ydata

            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()

            x_rescale = self.screen_size[0]/(xl[1]-xl[0])
            y_rescale = self.screen_size[1]/(yl[1]-yl[0])

            closest_dudes = [((self.x[ind]-x)*x_rescale)*((self.x[ind]-x)*x_rescale) + ((self.y[ind]-y)*y_rescale)*((self.y[ind]-y)*y_rescale) for ind in np.arange(0, len(self.x))]
            the_best_around_index = [np.argmin(closest_dudes[ind]) for ind in np.arange(0, len(self.x))]
            the_best_around = [closest_dudes[ind][ the_best_around_index[ind] ] for ind in np.arange(0, len(self.x))]
            which_line = np.argmin(the_best_around)
            index = the_best_around_index[which_line]

            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[which_line][index]
            y = self.y[which_line][index]

            ss = 0.025
            w = (xl[1]-xl[0]) * ss * self.screen_size[1]/self.screen_size[0]
            h = (yl[1]-yl[0]) * ss
            self.circ.set_width(w)
            self.circ.set_height(h)
            self.circ.xy = (x-0.5*w, y-0.5*h)
            point_color = self.point_colors[which_line][index]
            self.circ.set_edgecolor(point_color)

            xt = (x+1.5*w - xl[0])/(xl[1] - xl[0])
            yt = (y-0.5*h - yl[0])/(yl[1] - yl[0])

            if (xt < 0.5):
                self.text.set_horizontalalignment('left')
            else:
                self.text.set_horizontalalignment('right')
                xt = xt - 3*w/(xl[1] - xl[0])
                
            self.text.set_position((xt,yt))
            self.text.set_text(f'({x:.3g}, {y:.3g})')
            self.text.set_bbox(dict(edgecolor=point_color, facecolor=point_color, alpha=0.7))
            r, g, b = point_color[:3]
            self.text.set_color('black' if (0.299*r + 0.587*g + 0.114*b) > 0.6 else 'white')  # stay readable on light colors

            self.pos = [x, y]
            self.data_index = index
            self.which_line = which_line

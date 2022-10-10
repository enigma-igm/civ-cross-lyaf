import numpy as np
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.cm import register_cmap


def neut_cmap(s=0.5, n=256, name='NCcmap'):
    """Usage : neut_cmap(s=0.5, n=256)
    Output a color-map that is neutral (gray) in the middle,
    use reddish/warm colors at the upper end,
    and use blueish/cool colors at the lower end.

    Keywords:
      s = 0 < float < 1 : determines how quickly the color transition occurs.
      n = int : the number of color bins.
    """
    cdict = {'red':   ((0.0, 0.0, 0.0),  # x, value(x - epsilon), value(x + epsilon)
                       (.25, 0.0, 0.0),
                       (0.5,   s,   s),
                       (.75, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (.25, 1.0, 1.0),
                       (0.5,   s,   s),
                       (.75, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 1.0, 1.0),
                       (.25, 1.0, 1.0),
                       (0.5,   s,   s),
                       (.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0))}
    return LSC('NCcmap', cdict, n)


# NC = Neutral Center, cmap = color-map
NCcmap = neut_cmap()


for i in [NCcmap]:
    register_cmap(cmap=i)

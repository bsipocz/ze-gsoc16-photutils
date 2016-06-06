def _show_region(verts):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", lw=1)
    return patch

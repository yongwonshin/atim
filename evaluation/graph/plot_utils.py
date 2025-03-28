import matplotlib.colors as mcolors

colors = {
 "PrIM": '#fff191',
 "PrIMC": '#ffd100',
 'PS': '#ff8100',
 "SimplePIM": '#3daa8a',
 "ATiM": '#3b88ff',
 "CPU": "#aaabab",
 "CPU-Autotuned": "#aaabab",
 "CPU-PrIM": "#a0b0e0"
}

hatches = {
    "Kernel": "",
    "H2D": r"////////",
    "After": r"\\\\\\\\",
    "After Kernel": r"\\\\\\\\",
}

def adjust_color(color, factor=0.7):
    if isinstance(color, list):
        return [adjust_color(c, factor) for c in color]
    rgb = mcolors.to_rgb(color)
    adjusted_rgb = tuple([c * factor for c in rgb])
    return adjusted_rgb
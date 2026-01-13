import matplotlib as mpl
import shutil
from statsmodels.tsa.stattools import adfuller

def verify_stationarity(time_series):
    adf = adfuller(time_series)
    if adf[1] < 0.05:
        return 0
    
    else:
        time_series_differenced = time_series.diff().dropna()
        differenced_adf = adfuller(time_series_differenced)
        if differenced_adf[1] < 0.05:
            print("Os dados aparentam ser estacionários.")
            return 1
        
        else:
            print("Os dados aparentam ser não-estacionários. Os resultados do ARIMA deve ser tomados em consideração com cuidado.") 
            return 2
        
def set_plot_style():
    has_latex = shutil.which("latex") is not None

    if has_latex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{mathptmx}",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        })
    else:
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        })

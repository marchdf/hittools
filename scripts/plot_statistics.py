#!/usr/bin/env python3
"""
Plot the log files output from statistics.py
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================================================================
#
# Some defaults variables
#
# ===============================================================================
plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]

# ===============================================================================
#
# Function definitions
#
# ===============================================================================
def parse_log(resolution):
    """
    Parse the file written by statistics.py
    """
    pfx = "hit_ic_ut_"
    ext = ".log"
    fname = pfx + str(resolution) + ext
    with open(fname, "r") as f:
        for line in f:
            if " urms " in line:
                urms = float(line.split()[-1])
            if " KE " in line:
                KE = float(line.split()[6])
            if " lambda0 " in line:
                lambda0 = float(line.split()[-1])
            if " dilatation " in line:
                dilatation = float(line.split()[-1])

    return {
        "N": resolution ** 3,
        "urms": urms,
        "KE": KE,
        "lambda0": lambda0,
        "dilatation": dilatation,
    }


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Log files
    resolutions = [32, 64, 128, 256, 512, 768, 1024]

    # Now loop on files
    lst = []
    for k, resolution in enumerate(resolutions):
        lst.append(parse_log(resolution))

    df = pd.DataFrame(lst)

    # Normalizations
    urms_ut = np.sqrt(2)  # only for 256^3
    nu_ut = 0.0028
    epsilon_ut = 1.2
    lambda0_ut = np.sqrt(15 * nu_ut / epsilon_ut) * urms_ut
    tke_ut = 3 / 2 * urms_ut ** 2
    df["urms"] /= urms_ut
    df["lambda0"] /= lambda0_ut
    df["KE"] /= tke_ut
    df["dilatation"] /= (urms_ut / lambda0_ut) ** 2

    # Plot
    plt.figure(0)
    ax = plt.gca()
    plt.loglog(df.N, df.dilatation, color=cmap[0], marker=markertype[0], ms=10, lw=2)
    plt.xticks(
        (df.N[0], df.N[1], df.N[2], df.N[3], df.N[4], df.N[6]),
        (r"$32^3$", r"$64^3$", r"$128^3$", r"$256^3$", r"$512^3$", r"$1024^3$"),
    )
    plt.xlabel(r"$N$", fontsize=22, fontweight="bold")
    plt.ylabel(
        r"$\langle \theta_0 \theta_0 \rangle / (u_0' / \lambda_0)^2$",
        fontsize=22,
        fontweight="bold",
    )
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.gcf().subplots_adjust(bottom=0.14)
    plt.gcf().subplots_adjust(left=0.17)
    plt.savefig("dilatation_ic.png", format="png", dpi=300)
    plt.savefig("dilatation_ic.pdf", format="pdf", dpi=300)

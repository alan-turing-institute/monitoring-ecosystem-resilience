"""
Fill a dataframe of latitude, longitude, time, and "offset50", which is the difference
between the Euler Characteristic for the network of all signal pixels, and that for
the top 50% of signal pixels ordered by subgraph centrality.
"""

import os
import json
import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt

from subgraph_centrality import *

REGEX = "([\d]{1,3}\.[\d]+)_([\d]{1,3}\.[\d]+)_([\d]{4}-[0-1][\d]{1}-[0-3][\d]{1})"


def fill_data_frame_from_dir(input_dir, black_pix_signal=True, filename_filter_string=None):
    """
    Loop through all files in a directory, calculate EC50 for all of them,
    and fill a dataframe.
    If filename_filter_string is given, select only files
    """

    rows = []
    for filename in os.listdir(input_dir):
        if filename_filter_string:
            pass_filter = False
            for fs in filename_filter_string.split(","):
                if fs in filename:
                    pass_filter = True
                    break
            if not pass_filter:
                continue
        match = re.search(REGEX, filename)
        if match:
            longitude, latitude, time = match.groups()
        else:
            print("Unable to extract long, lat, time from filename")
            continue
        print("Processing {}".format(filename))
        img = read_image_file(os.path.join(input_dir, filename))
        try:
            feature_vec, sel_pix = subgraph_centrality(img,use_diagonal_neighbours=True, lower_threshold=True)
            offset50 = feature_vec[-1] - feature_vec[len(feature_vec)//2]
            row = {
                "longitude": longitude,
                "latitude": latitude,
                "time": datetime.strptime(time,"%Y-%m-%d"),
                "offset50": offset50
            }
            rows.append(row)
        except(ValueError):
            print("Issue with file {}".format(filename))
    df = pd.DataFrame()
    df = df.from_records(rows).sort_values(by="time").set_index("time")
    return df


def plot_time_series(df):
    fig, ax = plt.subplots()
    for key, grp in df.groupby(["latitude","longitude"]):
        ax = grp.plot(ax=ax, kind="line")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fill a dataframe from images in a directory")
    parser.add_argument("--input_dir",help="input directory",required=True)
    parser.add_argument("--black_signal",help="if set, treat black pixels as vegetation",action="store_true")
    parser.add_argument("--filter_string",help="comma-separated list of filter strings")
    args = parser.parse_args()
    black_signal = True if args.black_signal else False
    df = fill_data_frame_from_dir(args.input_dir, black_signal, args.filter_string)
#    plot_time_series(df)

import os
import datetime
import dateparser
import json
import re


def find_coords_string(file_path):
    """
    Parse a file path using a regular expresion to find a substring
    that looks like a set of coordinates,
    and return that.
    """

    match = re.search("([-]?[\d]{1,3}\.[\d]{1,3}[_,][-]?[\d]{1,3}\.[\d]{1,3})",file_path)
    if not match:
        return None
    coords_string = match.groups()[0]
    return coords_string


def find_mid_period(start_time, end_time):
    """
    Given two strings in the format YYYY-MM-DD return a
    string in the same format representing the middle (to
    the nearest day)
    """
    t0 = dateparser.parse(start_time)
    t1 = dateparser.parse(end_time)
    td = (t1 - t0).days
    mid = (t0 + timedelta(days=(td//2))).isoformat()
    return mid.split("T")[0]


def slice_time_period(start_date, end_date, n):
    """
    Divide the full period between the start_date and end_date into n equal-length
    (to the nearest day) chunks.
    Takes start_date and end_date as strings 'YYYY-MM-DD'.
    Returns a list of tuples
    [ (chunk0_start,chunk0_end),...]
    """
    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)
    if (not isinstance(start, datetime)) or (not isinstance(end, datetime)):
        raise RuntimeError("invalid time strings")
    td = end - start
    if td.days <= 0:
        raise RuntimeError("end_date must be after start_date")
    days_per_chunk = td.days // n
    output_list = []
    for i in range(n):
        chunk_start = start + timedelta(days=(i*days_per_chunk))
        chunk_end = start + timedelta(days=((i+1)*days_per_chunk))
        ## unless we are in the last chunk, which should finish at end_date
        if i == n-1:
            chunk_end = end
        output_list.append((chunk_start.isoformat().split("T")[0],
                           chunk_end.isoformat().split("T")[0]))
    return output_list


def get_num_n_day_slices(start_date, end_date, days_per_chunk):
    """
    Divide the full period between the start_date and end_date into n equal-length
    (to the nearest day) chunks. The size of the chunk is defined by days_per_chunk.
    Takes start_date and end_date as strings 'YYYY-MM-DD'.
    Returns an integer with the number of possible points avalaible in that time period]
    """
    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)
    if (not isinstance(start, datetime)) or (not isinstance(end, datetime)):
        raise RuntimeError("invalid time strings")
    td = end - start
    if td.days <= 0:
        raise RuntimeError("end_date must be after start_date")
    n = td.days//days_per_chunk

    return  n


def save_json(out_dict, output_dir, output_filename):
    """
    Given a dictionary, save
    to requested filename -
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w') as fp:
        json.dump(out_dict, fp, indent=2)

    print("Saved json file '{}'".format(output_path))

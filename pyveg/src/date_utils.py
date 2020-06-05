"""
Useful functions for manipulating dates and date strings,
e.g. splitting a period into sub-periods.

When dealing with date strings, ALWAYS use the ISO format YYYY-MM-DD

"""

import dateparser
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re


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


def slice_time_period_into_n(start_date, end_date, n):
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


def slice_time_period(start_date, end_date, period_length):
    """
    Slice a time period into chunks, whose length is determined by
    the period_length, which will be e.g. '30d' for 30 days,
    or '1m' for one month.

    Parameters
    ==========
    start_date: str, format YYYY-MM-DD
    end_date: str, format YYYY-MM-DD
    period_length: str, format '<integer><d|w|m|y>', e.g. 30d

    Returns
    =======
    periods: list of strings in format YYYY-MM-DD
    """
    periods = []
    start_datetime = datetime.fromisoformat(start_date)
    end_datetime = datetime.fromisoformat(end_date)

    # parse the period_length
    match = re.search("^([\d]+)([dwmy])", period_length)
    if not match:
        raise RuntimeError("Period length must be in format '<int><d|w|m|y>', e.g. 30d")

    num, units = match.groups()
    num = int(num)
    previous_date = start_datetime
    t = previous_date
    while True:
        if units == "d":
            t = previous_date + relativedelta(days=+num)
        elif units == "w":
            t = previous_date + relativedelta(weeks=+num)
        elif units == "m":
            t = previous_date + relativedelta(months=+num)
        else:
            t = previous_date + relativedelta(years=+num)
        # break out of the loop if we go after the end of our time period
        if t > end_datetime:
            break
        # otherwise, add this sub-period to the list
        periods.append([previous_date.isoformat().split("T")[0],
                        t.isoformat().split("T")[0]])
        previous_date = t
    return periods


def find_mid_period(start_time, end_time):
    """
    Given two strings in the format YYYY-MM-DD return a
    string in the same format representing the middle (to
    the nearest day)

    Parameters
    ==========
    start_time: str, date in format YYYY-MM-DD
    end_time: str, date in format YYYY-MM-DD

    Returns
    =======
    mid_date: str, mid point of those dates, format YYYY-MM-DD
    """
    t0 = dateparser.parse(start_time)
    t1 = dateparser.parse(end_time)
    td = (t1 - t0).days
    mid = (t0 + timedelta(days=(td//2))).isoformat()
    mid_date = mid.split("T")[0]
    return mid_date



def get_date_range_for_collection(date_range, coll_dict):
    """
    Return the intersection of the date range asked for by
    the user, and the min and max dates for that collection.

    Parameters
    ==========
    date_range: list or tuple of strings, format YYYY-MM-DD
    coll_dict: dictionary containing min_date and max_date kyes

    Returns
    =======
    tuple of strings, format YYYY-MM-DD
    """
    if not "min_date" in coll_dict.keys() or (not "max_date" in coll_dict.keys()):
        return date_range
    datetime_range = [dateparser.parse(d) for d in date_range]
    collection_min =  dateparser.parse(coll_dict["min_date"])
    collection_max =  dateparser.parse(coll_dict["max_date"])
    date_min = datetime_range[0] if datetime_range[0] > collection_min else collection_min
    date_max = datetime_range[1] if datetime_range[1] < collection_max else collection_max
    return (date_min.isoformat().split("T")[0], date_max.isoformat().split("T")[0])

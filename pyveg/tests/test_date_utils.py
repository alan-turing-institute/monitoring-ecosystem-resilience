"""
Test the utility functions that manipulate date strings
"""

from datetime import datetime

from pyveg.src.date_utils import *


def test_get_num_n_day_slices():

    start_date = "2015-01-01"
    end_date = "2016-01-01"
    days_per_chunk = 30
    assert get_num_n_day_slices(start_date, end_date, days_per_chunk) == 12


def test_slice_time_period_into_n():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for n_chunks in range(2,20):
        sub_periods = slice_time_period_into_n(start_date, end_date, n_chunks)
        assert isinstance(sub_periods, list)
        assert len(sub_periods) == n_chunks



def test_slice_time_period_by_days():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for i in range(2, 31):
        period_string = "{}d".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(start_date)
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(end_date)
        assert len(sub_periods) == 365 // i


def test_slice_time_period_by_weeks():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for i in range(1, 4):
        period_string = "{}w".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(start_date)
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(end_date)
        assert len(sub_periods) == 365 // (7*i)


def test_slice_time_period_by_months():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for i in range(1, 4):
        period_string = "{}m".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(start_date)
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(end_date)
        assert len(sub_periods) == 12 // i


def test_slice_time_period_by_years():
    start_date = "2015-01-01"
    end_date = "2020-01-01"
    for i in range(1, 4):
        period_string = "{}y".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(start_date)
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(end_date)
        assert len(sub_periods) == 5 // i

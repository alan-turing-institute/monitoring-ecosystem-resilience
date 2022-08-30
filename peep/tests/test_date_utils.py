"""
Test the utility functions that manipulate date strings
"""


from peep.src.date_utils import *


def test_get_num_n_day_slices():

    start_date = "2015-01-01"
    end_date = "2016-01-01"
    days_per_chunk = 30
    assert get_num_n_day_slices(start_date, end_date, days_per_chunk) == 12


def test_slice_time_period_into_n():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for n_chunks in range(2, 20):
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
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(
            start_date
        )
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(
            end_date
        )
        assert len(sub_periods) == 365 // i


def test_slice_time_period_by_weeks():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for i in range(1, 4):
        period_string = "{}w".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(
            start_date
        )
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(
            end_date
        )
        assert len(sub_periods) == 365 // (7 * i)


def test_slice_time_period_by_months():
    start_date = "2015-01-01"
    end_date = "2016-01-01"
    for i in range(1, 4):
        period_string = "{}m".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(
            start_date
        )
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(
            end_date
        )
        assert len(sub_periods) == 12 // i


def test_slice_time_period_by_years():
    start_date = "2015-01-01"
    end_date = "2020-01-01"
    for i in range(1, 4):
        period_string = "{}y".format(i)
        sub_periods = slice_time_period(start_date, end_date, period_string)
        assert isinstance(sub_periods, list)
        assert datetime.fromisoformat(sub_periods[0][0]) == datetime.fromisoformat(
            start_date
        )
        assert datetime.fromisoformat(sub_periods[-1][1]) <= datetime.fromisoformat(
            end_date
        )
        assert len(sub_periods) == 5 // i


def test_get_date_range_for_collection():
    date_range = ("2010-01-01", "2020-05-01")
    coll_dict = {"min_date": "2015-01-01", "max_date": "2019-01-01"}
    new_date_range = get_date_range_for_collection(date_range, coll_dict)
    assert new_date_range[0] == "2015-01-01"
    assert new_date_range[1] == "2019-01-01"


def test_assign_dates_to_tasks_more_dates_than_tasks():
    start_date = "2015-01-01"
    end_date = "2020-01-01"
    dates = slice_time_period(start_date, end_date, "1m")
    n_tasks = 20
    date_lists = assign_dates_to_tasks(dates, n_tasks)
    assert len(date_lists) == n_tasks
    for dl in date_lists:
        assert len(dl) == 3


def test_assign_dates_to_tasks_more_tasks_than_dates():
    start_date = "2015-01-01"
    end_date = "2020-01-01"
    dates = slice_time_period(start_date, end_date, "1m")
    n_tasks = 100
    date_lists = assign_dates_to_tasks(dates, n_tasks)
    assert len(date_lists) == len(dates)
    for dl in date_lists:
        assert len(dl) == 1


def test_get_time_diff():
    start_date = "1986-01-01"
    end_date = "2016-01-01"
    diff = get_time_diff(start_date, end_date)
    assert diff == -30

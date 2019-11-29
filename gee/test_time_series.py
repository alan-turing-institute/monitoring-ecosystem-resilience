"""
Tests for getting time-series of images
"""

import pytest
from download_images import *


def test_bad_timestring():
    """
    Give a nonsensical timestring and check we get an exception
    """
    with pytest.raises(RuntimeError) as err:
        chunks = divide_time_period("blah", "2015-05-12", 12)
        assert("invalid time strings" in str(err.value))


def test_end_before_start():
    """
    give an end time earlier than the start time
    """
    with pytest.raises(RuntimeError) as err:
        chunks = divide_time_period("blah", "2015-05-12", 12)
        assert("end date must be after" in str(err.value))


def test_slice_time_period_1():
    """
    Try dividing a year into 12.
    """
    t1 = "2018-01-01"
    t2 = "2019-01-01"
    chunks = divide_time_period(t1, t2, 12)
    assert(len(chunks)==12)
    assert(chunks[0][0] == "2018-01-01")
    assert(chunks[11][1] == "2019-01-01")
    assert("2018-11" in chunks[11][0])



def test_slice_time_period_2():
    """
    Try dividing a month into 3.
    """
    t1 = "2018-01-01"
    t2 = "2018-02-01"
    chunks = divide_time_period(t1, t2, 3)
    assert(len(chunks)==3)
    assert(chunks[0][0] == "2018-01-01")
    assert(chunks[1][0] == "2018-01-11")
    assert(chunks[2][1] == "2018-02-01")



def test_mid_time_1():
    """
    Given a couple of date strings, test
    we get the correct middle date
    """
    t1 = "2018-01-01"
    t2 = "2018-02-01"
    mid = find_mid_period(t1,t2)
    assert(mid=="2018-01-16")

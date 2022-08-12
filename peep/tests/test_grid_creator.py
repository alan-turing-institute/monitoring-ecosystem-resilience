import pytest
from icecream import ic
from matplotlib.pyplot import gci

import pyveg.src.grid_creator as grid_creator


def test_create_grid_of_chips():
    expected_bounding_box = [0, 0, 100, 100]

    actual_gdf = grid_creator.create_grid_of_chips(
        chip_px_width=10,
        pixel_scale=1,
        target_crs="EPSG:27700",
        bounding_box=expected_bounding_box,
    )

    assert len(actual_gdf) == 100

    assert all([a == e for a, e in zip(actual_gdf.total_bounds, expected_bounding_box)])


def test_get_chip_id():

    # Assign a shorter name for readability
    gcid = grid_creator.get_chip_id

    assert gcid(1, 2, 3) == "000001_0000002_003"
    assert gcid(123456789, 123456789, 12345) == "123456789_123456789_12345"


def test_get_image_id_from_chip():

    # Assign a shorter name for readability
    c2id = grid_creator.get_image_id_from_chip

    # Cases where only the width changes
    assert c2id("001536_0002048_032", 512) == "001536_0002048_512"
    assert c2id("001536_0002048_032", 512) == "001536_0002048_512"
    assert c2id("001536_0002048_016", 512) == "001536_0002048_512"
    # Cases where the `x` value changes
    assert c2id("001536_0002048_032", 1024) == "001024_0002048_1024"
    assert c2id("001536_0002048_016", 1024) == "001024_0002048_1024"
    # Cases there all three values change
    assert c2id("123456_1234567_032", 1024) == "122880_1233920_1024"
    assert c2id("123456_1234567_016", 1024) == "122880_1233920_1024"

    # This case should fail, because the requested chip size (16) is smaller than the source chip size (32)
    with pytest.raises(ValueError):
        c2id("123456_1234567_032", 16)


@pytest.mark.skip(reason="No suitable test data yet")
def test_coastline_to_poly():
    pass

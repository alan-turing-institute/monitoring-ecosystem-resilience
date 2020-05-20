import unittest
import pytest
import PIL


from pyveg.src.azure_utils import *

TEST_CONTAINER = "tests"


@unittest.skipIf(not os.path.exists(os.path.join(os.path.dirname(__file__),
                                                 "..", "azure_config.py")),
                 reason="Azure config not set")
def test_save_image():
    create_container(TEST_CONTAINER)
    # delete blob if it's already there
    blob_name = "white_image.png"
    delete_blob(blob_name, TEST_CONTAINER)
    img = PIL.Image.open(os.path.join(os.path.dirname(__file__),"..","testdata","white.png"))
    save_image(img, TEST_CONTAINER, blob_name, TEST_CONTAINER)
    assert check_blob_exists(blob_name, TEST_CONTAINER)
    delete_blob(blob_name, TEST_CONTAINER)



def test_read_image():
    create_container(TEST_CONTAINER)
    # delete blob if it's already there
    blob_name = "test_image.png"
    delete_blob(blob_name, TEST_CONTAINER)
    img = PIL.Image.open(os.path.join(os.path.dirname(__file__),"..","testdata","white.png"))
    save_image(img, TEST_CONTAINER, blob_name, TEST_CONTAINER)
    new_img = read_image(blob_name, TEST_CONTAINER)
    assert isinstance(new_img, PIL.PngImagePlugin.PngImageFile)


def test_save_json():
    create_container(TEST_CONTAINER)
    blob_name = "test.json"
    delete_blob(blob_name, TEST_CONTAINER)
    data = {"a":4,"b":5}
    save_json(data, "", blob_name, TEST_CONTAINER)
    assert check_blob_exists(blob_name, TEST_CONTAINER)
    delete_blob(blob_name, TEST_CONTAINER)


def test_read_json():
    create_container(TEST_CONTAINER)
    blob_name = "test_read.json"
    delete_blob(blob_name, TEST_CONTAINER)
    data = {"abc":44,"def":55}
    save_json(data, "", blob_name, TEST_CONTAINER)
    data = read_json(blob_name, TEST_CONTAINER)
    assert isinstance(data, dict)
    assert data["abc"]==44

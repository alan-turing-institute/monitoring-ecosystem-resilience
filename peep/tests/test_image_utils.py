"""
Test the functions in subgraph_centrality.py
"""

from peep.src.image_utils import *


def test_image_all_white():
    img = Image.open(
        os.path.join(os.path.dirname(__file__), "..", "testdata", "white.png")
    )
    assert image_all_same_colour(img, (255, 255, 255))
    assert not image_all_same_colour(img, (0, 0, 0))


def test_image_all_black():
    img = Image.open(
        os.path.join(os.path.dirname(__file__), "..", "testdata", "black.png")
    )
    assert image_all_same_colour(img, (0, 0, 0))
    assert not image_all_same_colour(img, (255, 255, 255))


def test_image_not_all_same():
    img = Image.open(
        os.path.join(
            os.path.dirname(__file__), "..", "testdata", "black_and_white_diagonal.png"
        )
    )
    assert not image_all_same_colour(img, (0, 0, 0))
    assert not image_all_same_colour(img, (255, 255, 255))


def test_compare_same_image():
    img1 = Image.open(
        os.path.join(
            os.path.dirname(__file__), "..", "testdata", "black_and_white_diagonal.png"
        )
    )
    img2 = Image.open(
        os.path.join(
            os.path.dirname(__file__), "..", "testdata", "black_and_white_diagonal.png"
        )
    )
    assert compare_binary_images(img1, img2) == 1.0


def test_compare_opposite_images():
    img1 = Image.open(
        os.path.join(
            os.path.dirname(__file__), "..", "testdata", "black_and_white_diagonal.png"
        )
    )
    img2 = Image.open(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "testdata",
            "black_and_white_diagonal_2.png",
        )
    )
    assert compare_binary_images(img1, img2) < 0.1


def test_pillow_to_numpy():
    img = Image.open(
        os.path.join(os.path.dirname(__file__), "..", "testdata", "white.png")
    )
    img_array = pillow_to_numpy(img)
    assert img_array.ndim == 2


def test_numpy_to_pillow():
    array = np.zeros((3, 3))
    pillow_image = numpy_to_pillow(array)
    rows, cols = pillow_image.size
    for row in range(rows):
        for col in range(cols):
            pixel = pillow_image.getpixel((row, col))
            assert pixel == 0


def test_hist_eq():
    img = (10 * np.random.randn(100, 100) + 255 // 2).astype("uint8")
    img = np.clip(img, 0, 255)
    img_eq = hist_eq(img)
    assert img.std() < img_eq.std()


def test_adaptive_threshold():
    img = (10 * np.random.randn(100, 100) + 255 // 2).astype("uint8")
    img = adaptive_threshold(img)

    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            is_binary = img[i, j] == 0 or img[i, j] == 255
            assert is_binary

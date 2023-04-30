import skimage
import numpy as np


def main():
    morphology()
    # heuristic()


def morphology():
    blue_threshold = 50
    object_min_size = 55
    object_connectivity = 2
    hole_area_threshold = 15

    image = skimage.io.imread("wshdchemung.jpg")

    output = skimage.morphology.remove_small_objects(
        threshold_mask(image, [81, 160, 248], threshold=blue_threshold),
        min_size=object_min_size,
        connectivity=object_connectivity,
    )

    output = skimage.morphology.remove_small_holes(
        output,
        area_threshold=hole_area_threshold,
    )

    # output = skimage.filters.gaussian(output, sigma=0.65)
    # output = skimage.filters.rank.mean(output, skimage.morphology.disk(10))

    skimage.io.imsave("output.jpg", np.invert(output))


def heuristic():
    image = skimage.io.imread("wshdchemung.jpg")

    # Mask out blue-ish text
    text_color = [59, 101, 121]
    text_threshold = 45
    text_mask = threshold_mask(image, text_color, threshold=text_threshold)
    image[text_mask] = [255, 0, 0]

    # Mask in blue-ish water
    water_color = [72, 148, 222]
    water_threshold = 40
    water_mask = np.invert(
        threshold_mask(image, water_color, threshold=water_threshold)
    )
    image[water_mask] = [0, 0, 0]

    non_zero_mask = np.logical_and(
        image[:, :, 0] != 0,
        image[:, :, 1] != 0,
        image[:, :, 2] != 0,
    )
    image[non_zero_mask] = [255, 255, 255]

    # image = skimage.filters.gaussian(image, sigma=0.65)
    # image = skimage.color.rgb2gray(image)

    min_size = 10
    connectivity = 1
    image = skimage.morphology.remove_small_objects(
        image != 0,
        min_size=min_size,
        connectivity=connectivity,
    )
    # image[small_objects_mask] = [0, 0, 0]

    skimage.io.imsave("output.jpg", image)


def threshold_mask(image, color, threshold=10):
    return np.logical_and(
        np.logical_and(
            color[0] - threshold < image[:, :, 0],
            image[:, :, 0] < color[0] + threshold,
        ),
        np.logical_and(
            color[1] - threshold < image[:, :, 1],
            image[:, :, 1] < color[1] + threshold,
        ),
        np.logical_and(
            color[2] - threshold < image[:, :, 2],
            image[:, :, 2] < color[2] + threshold,
        ),
    )


if __name__ == "__main__":
    main()

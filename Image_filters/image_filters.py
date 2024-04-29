import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 500 
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


def custom_convolution(img, kernel):
    pad_size = kernel.shape[0] // 2
    padded_img = np.pad(
        img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant"
    )
    filtered_img = np.zeros_like(img, np.float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                filtered_img[y, x, c] = np.sum(
                    padded_img[y : y + kernel.shape[0], x : x + kernel.shape[1], c]
                    * kernel
                )
    return filtered_img


def custom_median_blur(img, kernel_size):
    pad_size = kernel_size // 2
    padded_img = np.pad(
        img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant"
    )
    filtered_img = np.zeros_like(img, np.float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                values = padded_img[y : y + kernel_size, x : x + kernel_size, c]
                filtered_img[y, x, c] = np.median(values)
    return filtered_img


def custom_bilateral_filter(img, kernel_size, sigma_color, sigma_space):
    pad_size = kernel_size // 2
    padded_img = np.pad(
        img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant"
    )
    filtered_img = np.empty_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = padded_img[y : y + kernel_size, x : x + kernel_size]
            spatial_distances = np.sum((patch - img[y, x]) ** 2, axis=2) ** 0.5
            color_distances = np.sum((patch - img[y, x]) ** 2, axis=2)
            weights = np.exp(
                -spatial_distances / (2 * sigma_space**2)
                - color_distances / (2 * sigma_color**2)
            )
            filtered_img[y, x] = np.sum(
                patch * weights[:, :, np.newaxis], axis=(0, 1)
            ) / np.sum(weights)
    return filtered_img


def compare_filters(original_img, noisy_img, filtered_imgs, titles):
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    cv2.imwrite(os.path.join(images_dir, "original.png"), original_img)

    noisy_img = cv2.convertScaleAbs(noisy_img)
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
    plt.title("Noisy")
    cv2.imwrite(os.path.join(images_dir, "noisy.png"), noisy_img)

    for i, (filtered_img, title) in enumerate(zip(filtered_imgs, titles), start=4):
        filtered_img = cv2.convertScaleAbs(filtered_img)
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (MSE: {mse(original_img, filtered_img):.3f})")  # Dodano formatowanie do 3 miejsc po przecinku
        cv2.imwrite(os.path.join(images_dir, f"{title}.png"), filtered_img)
    plt.show()


def main():
    global images_dir

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_file_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    img = cv2.imread("image.jpg")
    noisy_img = add_noise(img)
    kernel_size = 5
    kernel_shape = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    print("Performing convolution filtering...")
    filtered_img_1 = custom_convolution(noisy_img, kernel_shape)
    mse_1 = mse(img, filtered_img_1)
    print(f"MSE for Convolution: {mse_1}")

    print("Performing median filtering...")
    filtered_img_2 = custom_median_blur(noisy_img, kernel_size)
    mse_2 = mse(img, filtered_img_2)
    print(f"MSE for Median Blur: {mse_2}")

    print("Performing bilateral filtering...")
    filtered_img_3 = custom_bilateral_filter(noisy_img, kernel_size, 75, 75)
    mse_3 = mse(img, filtered_img_3)
    print(f"MSE for Bilateral Filter: {mse_3}")

    print("Comparing filter results...")
    compare_filters(
        img,
        noisy_img,
        [filtered_img_1, filtered_img_2, filtered_img_3],
        ["Convolution", "Median Blur", "Bilateral Filter"],
    )

    min_mse = min([mse_1, mse_2, mse_3])
    if min_mse == mse_1:
        print("Convolution has the smallest MSE.")
    elif min_mse == mse_2:
        print("Median Blur has the smallest MSE.")
    else:
        print("Bilateral Filter has the smallest MSE.")

if __name__ == "__main__":
    main()
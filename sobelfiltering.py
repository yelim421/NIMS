import numpy as np
import matplotlib.pyplot as plt

def apply_filters(image, filters):
    H, W = image.shape

    output = np.zeros((H-2, W-2, len(filters)))

    for f, filter in enumerate(filters):
        for i in range(H-2):
            for j in range(W-2):
                output[i, j, f] = np.sum(image[i : i + 3, j : j + 3] * filter)

    return output


filters = [
    np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]),
    np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]),
    np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
]

image = plt.imread('image.jpg')

if len(image.shape) == 3:
    image = np.mean(image, axis=-1)

filtered_image = apply_filters(image, filters)

#시각화
fig, axes = plt.subplots(1, len(filters) + 1, figsize=(20, 20))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')

for i in range(len(filters)):
    axes[i+1].imshow(filtered_image[:, :, i], cmap='gray')
    axes[i+1].set_title(f'Filtered Image {i+1}')

for i in range(len(filters)):
    plt.imsave(f'filtered_image_{i+1}.png', filtered_image[:, :, i], cmap='gray')

plt.show()

"""
Module 3: Count black and white pixels, compute the percentage of white pixels
in .jpg images, and write results to a .csv file.
"""

import cv2
import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

FILENAMES = [
    r"../images/MASK_SK658 Llobe ch010039.jpg",
    r"../images/MASK_SK658 Slobe ch010066.jpg",
    r"../images/MASK_SK658 Slobe ch010147.jpg",
    r"../images/MASK_SK658 Slobe ch010110.jpg",
    r"../images/MASK_SK658 Slobe ch010130.jpg",
    r"../images/MASK_SK658 Slobe ch010114.jpg",
]

# Depths (microns) corresponding to each image above
DEPTHS = [15, 1000, 3000, 5300, 7000, 9900]

OUTPUT_CSV = "Percent_White_Pixels.csv"

# ── Image Processing ───────────────────────────────────────────────────────────

def analyze_image(filename):
    """
    Load a grayscale image, threshold it to binary, and return
    (white_count, black_count, white_percent).
    Raises FileNotFoundError if the image cannot be loaded.
    """
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {filename}")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    white = int(np.sum(binary == 255))
    black = int(np.sum(binary == 0))
    total = white + black
    white_pct = 100.0 * white / total if total > 0 else 0.0

    return white, black, white_pct


if len(FILENAMES) != len(DEPTHS):
    raise ValueError("FILENAMES and DEPTHS must be the same length.")

results = []

# ── Analyze each image ─────────────────────────────────────────────────────────
print("=" * 60)
print("Pixel counts per image")
print("=" * 60)

for filename, depth in zip(FILENAMES, DEPTHS):
    white, black, white_pct = analyze_image(filename)
    results.append({
        "Filename": filename,
        "Depth (microns)": depth,
        "White pixels": white,
        "Black pixels": black,
        "White percent": white_pct,
    })
    print(f"\n  File : {filename}")
    print(f"  Depth: {depth} microns")
    print(f"  White: {white:,} px  |  Black: {black:,} px  |  {white_pct:.2f}% white")

# ── Write CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print(f"Results written to '{OUTPUT_CSV}'")
print("=" * 60)


##############
# LECTURE 2: UNCOMMENT BELOW

# # Interpolate a point: given a depth, find the corresponding white pixel percentage

# interpolate_depth = float(input(colored(
#     "Enter the depth at which you want to interpolate a point (in microns): ", "yellow")))

# x = depths
# y = white_percents

# # You can also use 'quadratic', 'cubic', etc.
# i = interp1d(x, y, kind='linear')
# interpolate_point = i(interpolate_depth)
# print(colored(
#     f'The interpolated point is at the x-coordinate {interpolate_depth} and y-coordinate {interpolate_point}.', "green"))

# depths_i = depths[:]
# depths_i.append(interpolate_depth)
# white_percents_i = white_percents[:]
# white_percents_i.append(interpolate_point)


# # make two plots: one that doesn't contain the interpolated point, just the data calculated from your images, and one that also contains the interpolated point (shown in red)
# fig, axs = plt.subplots(2, 1)

# axs[0].scatter(depths, white_percents, marker='o', linestyle='-', color='blue')
# axs[0].set_title('Plot of depth of image vs percentage white pixels')
# axs[0].set_xlabel('depth of image (in microns)')
# axs[0].set_ylabel('white pixels as a percentage of total pixels')
# axs[0].grid(True)


# axs[1].scatter(depths_i, white_percents_i, marker='o',
#                linestyle='-', color='blue')
# axs[1].set_title(
#     'Plot of depth of image vs percentage white pixels with interpolated point (in red)')
# axs[1].set_xlabel('depth of image (in microns)')
# axs[1].set_ylabel('white pixels as a percentage of total pixels')
# axs[1].grid(True)
# axs[1].scatter(depths_i[len(depths_i)-1], white_percents_i[len(white_percents_i)-1],
#                color='red', s=100, label='Highlighted point')


# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

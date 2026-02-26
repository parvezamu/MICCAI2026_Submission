import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
import os

# =========================
# CONFIG
# =========================
NPZ_PATH = "/home/pahm409/preprocessed_isles_dual_v2/sub-strokecase0007.npz"  # <-- change
PATCH = (96, 96, 96)
SEED = 123  # fixed seed so you see consistent mismatch
OUT_DIR = "./aug_debug_outputs"  # output folder for saved PNGs

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def center_patch(arr, patch=(96, 96, 96)):
    D, H, W = arr.shape
    pd, ph, pw = patch
    d0 = max(0, D // 2 - pd // 2)
    h0 = max(0, H // 2 - ph // 2)
    w0 = max(0, W // 2 - pw // 2)
    out = arr[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]

    # pad if needed
    pad = [(0, patch[i] - out.shape[i]) for i in range(3)]
    pad = [(p0, max(0, p1)) for (p0, p1) in pad]
    if any(p[1] > 0 for p in pad):
        out = np.pad(out, pad, mode="constant", constant_values=0)
    return out

def plot_overlay(ax, img, mask, title):
    ax.imshow(img, cmap="gray")
    ax.contour(mask.astype(float), levels=[0.5], linewidths=1)
    ax.set_title(title)
    ax.axis("off")

def save_overlay(img, mask, path, title):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.contour(mask.astype(float), levels=[0.5], linewidths=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _center_crop_or_pad(arr, target_shape, cval=0):
    out = arr

    # crop
    for ax in range(3):
        if out.shape[ax] > target_shape[ax]:
            start = (out.shape[ax] - target_shape[ax]) // 2
            sl = [slice(None)] * 3
            sl[ax] = slice(start, start + target_shape[ax])
            out = out[tuple(sl)]

    # pad
    pad_width = []
    for ax in range(3):
        diff = target_shape[ax] - out.shape[ax]
        if diff > 0:
            p0 = diff // 2
            p1 = diff - p0
            pad_width.append((p0, p1))
        else:
            pad_width.append((0, 0))
    if any(p != (0, 0) for p in pad_width):
        out = np.pad(out, pad_width, mode="constant", constant_values=cval)

    return out

# =========================
# YOUR ORIGINAL AUGMENTATION (BAD)
# =========================
def bad_apply_augmentations(dwi, adc, mask):
    info = {}

    # rotation (BAD: different random params)
    if np.random.rand() > 0.5:
        angle1 = np.random.uniform(-15, 15)
        axes_list = [(0, 1), (0, 2), (1, 2)]
        axes1 = axes_list[np.random.randint(0, 3)]

        dwi = rotate(dwi, angle1, axes=axes1, reshape=False, order=1, mode="constant", cval=0)
        mask = rotate(mask, angle1, axes=axes1, reshape=False, order=0, mode="constant", cval=0)

        angle2 = np.random.uniform(-15, 15)
        axes2 = axes_list[np.random.randint(0, 3)]

        adc = rotate(adc, angle2, axes=axes2, reshape=False, order=1, mode="constant", cval=0)

        info["rot_dwi_angle"] = angle1
        info["rot_dwi_axes"] = axes1
        info["rot_adc_angle"] = angle2
        info["rot_adc_axes"] = axes2
    else:
        info["rot"] = "skipped"

    # elastic (BAD: different random fields)
    if np.random.rand() > 0.7:
        shape = dwi.shape

        dx1 = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        dy1 = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        dz1 = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        ind1 = (np.reshape(x + dx1, (-1, 1)), np.reshape(y + dy1, (-1, 1)), np.reshape(z + dz1, (-1, 1)))

        dwi = map_coordinates(dwi, ind1, order=1, mode="reflect").reshape(shape)
        mask = map_coordinates(mask, ind1, order=0, mode="constant", cval=0).reshape(shape)

        # new random field for ADC (bug)
        dx2 = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        dy2 = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        dz2 = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        ind2 = (np.reshape(x + dx2, (-1, 1)), np.reshape(y + dy2, (-1, 1)), np.reshape(z + dz2, (-1, 1)))

        adc = map_coordinates(adc, ind2, order=1, mode="reflect").reshape(shape)

        info["elastic"] = "applied (DIFFERENT fields for DWI vs ADC)"
    else:
        info["elastic"] = "skipped"

    # scaling (BAD: different random scales)
    if np.random.rand() > 0.5:
        s1 = np.random.uniform(0.9, 1.1)
        dwi_s = zoom(dwi, s1, order=1, mode="constant", cval=0)
        mask_s = zoom(mask, s1, order=0, mode="constant", cval=0)
        dwi = _center_crop_or_pad(dwi_s, dwi.shape, cval=0)
        mask = _center_crop_or_pad(mask_s, mask.shape, cval=0)

        s2 = np.random.uniform(0.9, 1.1)
        adc_s = zoom(adc, s2, order=1, mode="constant", cval=0)
        adc = _center_crop_or_pad(adc_s, adc.shape, cval=0)

        info["scale_dwi"] = s1
        info["scale_adc"] = s2
    else:
        info["scale"] = "skipped"

    # flip (GOOD: shared)
    if np.random.rand() > 0.5:
        axis = np.random.randint(0, 3)
        dwi = np.flip(dwi, axis=axis).copy()
        adc = np.flip(adc, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        info["flip_axis"] = axis
    else:
        info["flip"] = "skipped"

    return dwi, adc, mask, info

# =========================
# FIXED AUGMENTATION (ALIGNED)
# =========================
def aligned_apply_augmentations(dwi, adc, mask):
    info = {}

    # shared rotation
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)
        axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
        dwi = rotate(dwi, angle, axes=axes, reshape=False, order=1, mode="constant", cval=0)
        adc = rotate(adc, angle, axes=axes, reshape=False, order=1, mode="constant", cval=0)
        mask = rotate(mask, angle, axes=axes, reshape=False, order=0, mode="constant", cval=0)
        info["rot_angle"] = angle
        info["rot_axes"] = axes
    else:
        info["rot"] = "skipped"

    # shared elastic
    if np.random.rand() > 0.7:
        shape = dwi.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), 3) * 10

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
        ind = (np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1)))

        dwi = map_coordinates(dwi, ind, order=1, mode="reflect").reshape(shape)
        adc = map_coordinates(adc, ind, order=1, mode="reflect").reshape(shape)
        mask = map_coordinates(mask, ind, order=0, mode="constant", cval=0).reshape(shape)
        info["elastic"] = "applied (SAME field)"
    else:
        info["elastic"] = "skipped"

    # shared scaling
    if np.random.rand() > 0.5:
        s = np.random.uniform(0.9, 1.1)

        dwi_s = zoom(dwi, s, order=1, mode="constant", cval=0)
        adc_s = zoom(adc, s, order=1, mode="constant", cval=0)
        mask_s = zoom(mask, s, order=0, mode="constant", cval=0)

        dwi = _center_crop_or_pad(dwi_s, dwi.shape, cval=0)
        adc = _center_crop_or_pad(adc_s, adc.shape, cval=0)
        mask = _center_crop_or_pad(mask_s, mask.shape, cval=0)

        info["scale"] = s
    else:
        info["scale"] = "skipped"

    # shared flip
    if np.random.rand() > 0.5:
        axis = np.random.randint(0, 3)
        dwi = np.flip(dwi, axis=axis).copy()
        adc = np.flip(adc, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        info["flip_axis"] = axis
    else:
        info["flip"] = "skipped"

    return dwi, adc, mask, info

# =========================
# MAIN
# =========================
def main():
    np.random.seed(SEED)

    data = np.load(NPZ_PATH)
    dwi = center_patch(data["dwi"], PATCH)
    adc = center_patch(data["adc"], PATCH)
    m = center_patch(data["mask"], PATCH)

    z = PATCH[2] // 2

    # Apply BAD
    np.random.seed(SEED)
    dwi_b, adc_b, m_b, info_b = bad_apply_augmentations(dwi.copy(), adc.copy(), m.copy())

    # Apply ALIGNED
    np.random.seed(SEED)
    dwi_g, adc_g, m_g, info_g = aligned_apply_augmentations(dwi.copy(), adc.copy(), m.copy())

    print("\n================ BAD AUG (your current logic) ================")
    for k, v in info_b.items():
        print(f"{k}: {v}")

    print("\n================ ALIGNED AUG (fixed) ================")
    for k, v in info_g.items():
        print(f"{k}: {v}")

    # Save individual images
    save_overlay(dwi[:, :, z], m[:, :, z] > 0, f"{OUT_DIR}/before_dwi.png", "DWI BEFORE")
    save_overlay(adc[:, :, z], m[:, :, z] > 0, f"{OUT_DIR}/before_adc.png", "ADC BEFORE")

    save_overlay(dwi_b[:, :, z], m_b[:, :, z] > 0, f"{OUT_DIR}/after_bad_dwi.png", "DWI AFTER BAD")
    save_overlay(adc_b[:, :, z], m_b[:, :, z] > 0, f"{OUT_DIR}/after_bad_adc.png", "ADC AFTER BAD")

    save_overlay(dwi_g[:, :, z], m_g[:, :, z] > 0, f"{OUT_DIR}/after_aligned_dwi.png", "DWI AFTER ALIGNED")
    save_overlay(adc_g[:, :, z], m_g[:, :, z] > 0, f"{OUT_DIR}/after_aligned_adc.png", "ADC AFTER ALIGNED")

    print(f"\nSaved individual images to: {OUT_DIR}")

    # Save combined grid
    plt.figure(figsize=(12, 8))

    ax = plt.subplot(3, 2, 1); plot_overlay(ax, dwi[:, :, z], m[:, :, z] > 0, "DWI BEFORE")
    ax = plt.subplot(3, 2, 2); plot_overlay(ax, adc[:, :, z], m[:, :, z] > 0, "ADC BEFORE")

    ax = plt.subplot(3, 2, 3); plot_overlay(ax, dwi_b[:, :, z], m_b[:, :, z] > 0, "DWI AFTER BAD")
    ax = plt.subplot(3, 2, 4); plot_overlay(ax, adc_b[:, :, z], m_b[:, :, z] > 0, "ADC AFTER BAD")

    ax = plt.subplot(3, 2, 5); plot_overlay(ax, dwi_g[:, :, z], m_g[:, :, z] > 0, "DWI AFTER ALIGNED")
    ax = plt.subplot(3, 2, 6); plot_overlay(ax, adc_g[:, :, z], m_g[:, :, z] > 0, "ADC AFTER ALIGNED")

    plt.tight_layout()
    grid_path = f"{OUT_DIR}/comparison_grid.png"
    plt.savefig(grid_path, dpi=200)
    plt.close()

    print(f"Saved grid image to: {grid_path}\n")

if __name__ == "__main__":
    main()


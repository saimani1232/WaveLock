"""
Make the mobile app's logo and icons from the source logo
(mobile/branding/wavelock-logo-source.png, square, opaque dark background):

    python tools/make_icons.py

Writes mobile/src/assets/:
  icon-192.png, icon-512.png       home-screen icons ("any"), full bleed
  icon-maskable-512.png            Android shaped icons: the logo inside the
                                   central 80 % safe zone, on its own background colour
  icon-180.png                     iPhone home screen (apple-touch-icon)
  icon-48.png                      browser tab (favicon)
  logo-128.png, logo-256.png       the mark in the top bar / start-up screen
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOBILE = os.path.dirname(HERE)
SOURCE = os.path.join(MOBILE, "branding", "wavelock-logo-source.png")
OUT = os.path.join(MOBILE, "src", "assets")


def resized(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def main():
    img = cv2.imread(SOURCE, cv2.IMREAD_COLOR)
    if img is None or img.shape[0] != img.shape[1]:
        raise SystemExit(f"{SOURCE}: expected a square image")
    os.makedirs(OUT, exist_ok=True)
    for name, size in (("icon-192.png", 192), ("icon-512.png", 512), ("icon-180.png", 180),
                       ("icon-48.png", 48), ("logo-128.png", 128), ("logo-256.png", 256)):
        cv2.imwrite(os.path.join(OUT, name), resized(img, size))
        print("wrote", name)
    # maskable: the whole logo within the 80 % safe zone, padded with the
    # source's own border colour (median of its outermost pixels)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    colour = np.median(border, axis=0).astype(np.uint8)
    canvas = np.empty((512, 512, 3), np.uint8)
    canvas[:] = colour
    inner = int(round(512 * 0.80))
    off = (512 - inner) // 2
    logo = resized(img, inner).astype(np.float32)
    # feather the outer 12 % of the logo into the padding (no visible square edge)
    ramp = np.minimum(np.arange(inner), np.arange(inner)[::-1]) / (0.12 * inner)
    edge = np.clip(ramp, 0, 1)
    alpha = np.minimum.outer(edge, edge)
    alpha = (alpha * alpha * (3 - 2 * alpha))[:, :, None]          # smoothstep
    region = canvas[off:off + inner, off:off + inner].astype(np.float32)
    canvas[off:off + inner, off:off + inner] = (alpha * logo + (1 - alpha) * region).round().astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "icon-maskable-512.png"), canvas)
    print("wrote icon-maskable-512.png (background", "#%02x%02x%02x" % tuple(int(c) for c in colour[::-1]) + ")")


if __name__ == "__main__":
    main()

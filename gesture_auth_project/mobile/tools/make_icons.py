"""
Render the app icons (mobile/src/assets/icon-*.png) from web/assets/logo.svg
with headless Chrome. Run once when the logo changes:

    python tools/make_icons.py
"""
import base64
import os

from selenium import webdriver

HERE = os.path.dirname(os.path.abspath(__file__))
MOBILE = os.path.dirname(HERE)
SVG = os.path.join(os.path.dirname(MOBILE), "web", "assets", "logo.svg")
OUT = os.path.join(MOBILE, "src", "assets")
BG = "#0b1118"

JS = """
const [src, size, pad, bg] = arguments;
const done = arguments[arguments.length - 1];
const img = new Image();
img.onload = () => {
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d');
  g.fillStyle = bg; g.fillRect(0, 0, size, size);
  const inner = size * (1 - 2 * pad);
  g.drawImage(img, (size - inner) / 2, (size - inner) / 2, inner, inner);
  done(c.toDataURL('image/png'));
};
img.onerror = () => done(null);
img.src = src;
"""


def main():
    svg = open(SVG, "rb").read()
    src = "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    d = webdriver.Chrome(options=opts)
    try:
        d.get("about:blank")
        os.makedirs(OUT, exist_ok=True)
        # "any" icons: small margin; maskable: the logo inside the 80 % safe zone
        for name, size, pad in (("icon-192.png", 192, 0.08), ("icon-512.png", 512, 0.08),
                                ("icon-maskable-512.png", 512, 0.20)):
            url = d.execute_async_script(JS, src, size, pad, BG)
            if not url:
                raise SystemExit("could not render the logo")
            with open(os.path.join(OUT, name), "wb") as f:
                f.write(base64.b64decode(url.split(",", 1)[1]))
            print("wrote", name)
    finally:
        d.quit()


if __name__ == "__main__":
    main()

# Web-ready brand art

The app loads its images from this folder, never from `visuals/` directly.

`BID_Logo_Badge-TXT.png` and `(BID_Logo_Badge.png` are Adobe **PSD** documents
saved with a `.png` extension (magic bytes `8BPS`). Browsers and Streamlit
cannot render them, and the `-TXT` one is CMYK, so Pillow reads its colors
inverted. The `ICON_*` and `Chatbot_Logo_002` files *are* real PNGs, but at
1500x1500 they are far larger than any place they appear.

| File | Source | Used for |
|---|---|---|
| `bid_logo_horizontal.png` (1000x437) | `BID_Logo_Badge-TXT.png` | `st.logo()` — top-left lockup |
| `bid_badge.png` (512x512) | `(BID_Logo_Badge.png` | favicon + collapsed-sidebar icon |
| `icon_about.png` (96x96) | `ICON_About.png` | About expander |
| `icon_change_password.png` (96x96) | `ICON_Change_Password.png` | Change Password expander |
| `icon_log_out.png` (96x96) | `ICON_Log_Out.png` | Log Out button |
| `icon_question.png` (96x96) | `ICON_Question.png` | Example Questions heading |

The four `icon_*` files keep the artwork's full square canvas (no alpha trim)
so the set stays visually consistent; the two lockups are trimmed to their
alpha bounding box.

`visuals/Chatbot_Logo_002.png` is currently unused — it briefly ran as a
wordmark above the chat page title and was dropped, leaving the top-left
lockup as the only brand mark.

Regenerate after editing the originals. macOS `sips` handles the PSD CMYK->RGB
conversion correctly; Pillow alone does not:

```bash
sips -s format png 'visuals/BID_Logo_Badge-TXT.png' --out /tmp/txt.png
sips -s format png 'visuals/(BID_Logo_Badge.png'    --out /tmp/badge.png
python - <<'PY'
from PIL import Image

def prep(src, out, box, trim=True):
    im = Image.open(src).convert("RGBA")
    if trim:
        im = im.crop(im.getchannel("A").getbbox())
    im.thumbnail(box, Image.LANCZOS)
    im.save(out, "PNG", optimize=True)

prep("/tmp/txt.png",   "visuals/web/bid_logo_horizontal.png", (1000, 1000))
prep("/tmp/badge.png", "visuals/web/bid_badge.png",           (512, 512))
for src, out in [("ICON_About", "icon_about"),
                 ("ICON_Change_Password", "icon_change_password"),
                 ("ICON_Log_Out", "icon_log_out"),
                 ("ICON_Question", "icon_question")]:
    prep(f"visuals/{src}.png", f"visuals/web/{out}.png", (96, 96), trim=False)
PY
```

## Adding another PNG icon to a widget

`st.button` / `st.expander` accept only emoji or `:material/*` names for
`icon=`, so PNG icons go in as a `::before` background keyed on the widget's
`st-key-<key>` class. Give the widget a `key`, then add a
`_widget_icon_css(key, path, target=...)` line in `frontend_streamlit.py`.
`target` must name the element holding the label — `"button"` or `"summary"`
(expander header). Without it the rule matches every markdown paragraph in the
container and stamps the icon on each child widget's label too.

## Where there is no PNG

The sidebar nav, New Chat / Delete Current Chat, System Status, and the
"In development" card captions use Streamlit's built-in Material icons
(`:material/forum:`, `:material/add:`, ...) tinted to `BID_TEAL` (`#00a79d`,
the BID brand teal) so they sit with the PNG set. Swapping any of
them for a real PNG later is a one-line change.

Note the two different DOM shapes: an `icon=` widget parameter renders as
`[data-testid="stIconMaterial"]`, while `:material/x:` written inside markdown
renders as a bare `span[role="img"]`. The tint rule has to cover both. The
success/error alerts are deliberately left out of it — their green and red are
semantic.

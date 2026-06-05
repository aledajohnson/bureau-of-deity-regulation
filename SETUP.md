# Airheads — Smart Mirror Setup

## First-time install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (development)

```bash
.venv/bin/python mirror.py
```

## Run (event day)

```bash
./launcher.sh
```

Or double-click `launcher.sh` in Finder (right-click → Open if macOS blocks it the first time).

---

## Controls

| Input | Action |
|---|---|
| **Space / Enter / USB button** | Start sequence (idle) or reset (mid-sequence) |
| `↑ ↓` | Adjust wig vertical position — value prints to terminal |
| `← →` | Adjust wig width scale — value prints to terminal |
| `Esc` | Quit |

Any USB button that sends Space or Enter works out of the box — no configuration needed.

## Wig calibration (do this before the event)

1. Run the mirror with someone standing at the intended distance
2. Use arrow keys to dial in position and scale until it sits correctly
3. Copy the printed values into `mirror.py`:
   ```python
   WIG_WIDTH_SCALE = 1.5   # ← your tuned value
   WIG_RING_BOTTOM = 0.55  # ← your tuned value
   ```

## Assets

- `assets/wig.png` — transparent RGBA PNG, cropped tight to content

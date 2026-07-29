"""
Generates a hyper-realistic snake (gradient-shaded, tapered, scaled body,
eyes + flicking tongue) that eats its way across the real GitHub contribution
graph, output as two theme-matched, self-looping SMIL-animated SVGs.

Env vars:
  GITHUB_TOKEN     - provided automatically by Actions (needs no extra scopes;
                      contributionCalendar is public data on the profile)
  SNAKE_USERNAME   - your GitHub username (defaults to GITHUB_REPOSITORY_OWNER)
"""
import os
import json
import numpy as np
import requests

# ---------------------------------------------------------------- geometry --
CELL = 15
GRID_LEFT = 34
GRID_TOP = 26
SEG_LEN = 10
HEAD_WIDTH = 10.5
TAIL_WIDTH = 1.4
SAMPLES_PER_SEG = 3
DUR = 34.0
STRIDE = 2

def cell_center(week, day):
    return (GRID_LEFT + week * CELL + CELL / 2, GRID_TOP + day * CELL + CELL / 2)

def boustrophedon_path(n_weeks):
    path = []
    for w in range(n_weeks):
        days = range(7) if w % 2 == 0 else range(6, -1, -1)
        for d in days:
            path.append((w, d))
    return path

def catmull_rom(points, samples_per_seg=SAMPLES_PER_SEG):
    pts = np.array(points, dtype=float)
    if len(pts) < 2:
        return pts
    pts = np.vstack([pts[0] + (pts[0] - pts[1]), pts, pts[-1] + (pts[-1] - pts[-2])])
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            out.append((x, y))
    out.append(tuple(pts[-2]))
    return np.array(out)

def taper_profile(n, head_w=HEAD_WIDTH, tail_w=TAIL_WIDTH):
    t = np.linspace(0, 1, n)
    base = head_w + (tail_w - head_w) * (t ** 1.35)
    bulge = 1.0 + 0.16 * np.exp(-((t - 0.16) ** 2) / (2 * 0.05 ** 2))
    w = base * bulge
    tip = np.clip((1 - t) / 0.05, 0, 1)
    return np.where(t > 0.95, w * (0.3 + 0.7 * tip), w)

def offset_ribbon(curve, widths):
    n = len(curve)
    tang = np.zeros_like(curve)
    tang[1:-1] = curve[2:] - curve[:-2]
    tang[0] = curve[1] - curve[0]
    tang[-1] = curve[-1] - curve[-2]
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    norm[norm == 0] = 1
    tang = tang / norm
    perp = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    right = curve + perp * widths[:, None]
    left = curve - perp * widths[:, None]
    return np.vstack([right, left[::-1]])

def path_d(poly):
    return "M" + "L".join(f"{x:.0f},{y:.0f}" for x, y in poly) + "Z"

def build_window(centers, head_idx, seg_len=SEG_LEN):
    start = max(0, head_idx - seg_len)
    window = centers[start:head_idx + 1][::-1]
    while len(window) < seg_len + 1:
        window.append(window[-1])
    return window

def frame_polygon_d(centers, head_idx):
    window = build_window(centers, head_idx)
    curve = catmull_rom(window)
    widths = taper_profile(len(curve)) / 2.0
    return path_d(offset_ribbon(curve, widths)), curve, widths

# ------------------------------------------------------------------ data ---
def fetch_contributions(username, token):
    query = """
    query($login: String!) {
      user(login: $login) {
        contributionsCollection {
          contributionCalendar {
            weeks { contributionDays { weekday contributionCount } }
          }
        }
      }
    }"""
    r = requests.post(
        "https://api.github.com/graphql",
        json={"query": query, "variables": {"login": username}},
        headers={"Authorization": f"bearer {token}"},
        timeout=30,
    )
    r.raise_for_status()
    weeks = r.json()["data"]["user"]["contributionsCollection"]["contributionCalendar"]["weeks"]
    counts = [[d["contributionCount"] for d in wk["contributionDays"]] for wk in weeks]
    flat = [c for wk in counts for c in wk if c > 0]
    p75 = np.percentile(flat, 75) if flat else 1
    p50 = np.percentile(flat, 50) if flat else 1
    p25 = np.percentile(flat, 25) if flat else 1

    def level(c):
        if c == 0:
            return 0
        if c <= max(1, p25):
            return 1
        if c <= max(2, p50):
            return 2
        if c <= max(3, p75):
            return 3
        return 4

    grid = np.zeros((len(counts), 7), dtype=int)
    for w, wk in enumerate(counts):
        for d, c in enumerate(wk):
            grid[w, d] = level(c)
    return grid

# ------------------------------------------------------------------- svg ---
PALETTES = {
    "dark": dict(bg="#0A101F", empty="#161B2E",
                 level=["#161B2E", "#0E4430", "#12633F", "#17924C", "#1DBE5C"],
                 body_a="#34D399", belly="#6EE7B7", body_b="#065F46", scale="#047857",
                 eye="#F3F4F6", pupil="#0A101F", tongue="#EF4444"),
    "light": dict(bg="#F3F5FA", empty="#E4E9F5",
                  level=["#E4E9F5", "#BCE8D2", "#7FD4A6", "#3FB578", "#0E8F52"],
                  body_a="#059669", belly="#34D399", body_b="#064E3B", scale="#047857",
                  eye="#101828", pupil="#F3F5FA", tongue="#DC2626"),
}

def build_svg(mode, data):
    pal = PALETTES[mode]
    n_weeks = data.shape[0]
    canvas_w = GRID_LEFT + n_weeks * CELL + 14
    canvas_h = GRID_TOP + 7 * CELL + 14
    path = boustrophedon_path(n_weeks)
    centers = [cell_center(w, d) for w, d in path]
    n_steps = len(path)
    step_dur = DUR / n_steps

    svg = [f'<svg width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect width="{canvas_w}" height="{canvas_h}" rx="6" fill="{pal["bg"]}"/>')
    svg.append(f'''<defs>
<linearGradient id="bodyGrad-{mode}" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0%" stop-color="{pal["body_a"]}"/>
  <stop offset="42%" stop-color="{pal["belly"]}"/>
  <stop offset="100%" stop-color="{pal["body_b"]}"/>
</linearGradient>
<pattern id="scalePat-{mode}" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
  <circle cx="2" cy="2" r="1.6" fill="none" stroke="{pal["scale"]}" stroke-width="0.55" opacity="0.55"/>
</pattern>
</defs>''')

    for idx, (w, d) in enumerate(path):
        level = int(data[w, d])
        x, y = GRID_LEFT + w * CELL, GRID_TOP + d * CELL
        color = pal["level"][level]
        eat_time = idx * step_dur
        svg.append(
            f'<rect x="{x+1.2:.1f}" y="{y+1.2:.1f}" width="{CELL-2.4}" height="{CELL-2.4}" rx="2.5" fill="{color}">'
            f'<animate attributeName="fill" begin="{eat_time:.3f}s" dur="{DUR}s" repeatCount="indefinite" '
            f'calcMode="discrete" values="{color};{pal["empty"]};{color}" '
            f'keyTimes="0;{(eat_time+0.01)/DUR:.4f};0.999"/></rect>'
        )

    frame_idxs = list(range(0, n_steps, STRIDE))
    if frame_idxs[-1] != n_steps - 1:
        frame_idxs.append(n_steps - 1)
    d_values = [frame_polygon_d(centers, i)[0] for i in frame_idxs]
    kt = ";".join(f"{i/(n_steps-1):.4f}" for i in frame_idxs)

    svg.append(f'<path id="snakeBody-{mode}" fill="url(#bodyGrad-{mode})" stroke="{pal["body_b"]}" stroke-width="0.6" '
               f'd="{d_values[0]}"><animate attributeName="d" dur="{DUR}s" repeatCount="indefinite" '
               f'calcMode="linear" keyTimes="{kt}" values="{";".join(d_values)}"/></path>')
    svg.append(f'<use href="#snakeBody-{mode}" fill="url(#scalePat-{mode})" opacity="0.7"/>')

    eye_x0, eye_y0, eye_x1, eye_y1 = [], [], [], []
    head_pts, head_dirs = [], []
    for i in frame_idxs:
        curve = catmull_rom(build_window(centers, i))
        widths = taper_profile(len(curve)) / 2.0
        hx, hy = curve[0]
        hd = curve[0] - curve[3]
        nrm = np.linalg.norm(hd)
        hd = hd / nrm if nrm else np.array([1.0, 0.0])
        perp = np.array([-hd[1], hd[0]])
        w0 = widths[0]
        fwd = -hd * (w0 * 0.15)
        eye_x0.append(hx + perp[0]*w0*0.55 + fwd[0]); eye_y0.append(hy + perp[1]*w0*0.55 + fwd[1])
        eye_x1.append(hx - perp[0]*w0*0.55 + fwd[0]); eye_y1.append(hy - perp[1]*w0*0.55 + fwd[1])
        head_pts.append((hx, hy)); head_dirs.append(hd)

    for gx, gy in [(eye_x0, eye_y0), (eye_x1, eye_y1)]:
        cxs = ";".join(f"{v:.0f}" for v in gx)
        cys = ";".join(f"{v:.0f}" for v in gy)
        svg.append(f'<circle r="1.5" fill="{pal["eye"]}" cx="{gx[0]:.0f}" cy="{gy[0]:.0f}">'
                   f'<animate attributeName="cx" dur="{DUR}s" repeatCount="indefinite" keyTimes="{kt}" values="{cxs}"/>'
                   f'<animate attributeName="cy" dur="{DUR}s" repeatCount="indefinite" keyTimes="{kt}" values="{cys}"/></circle>')
        svg.append(f'<circle r="0.75" fill="{pal["pupil"]}" cx="{gx[0]:.0f}" cy="{gy[0]:.0f}">'
                   f'<animate attributeName="cx" dur="{DUR}s" repeatCount="indefinite" keyTimes="{kt}" values="{cxs}"/>'
                   f'<animate attributeName="cy" dur="{DUR}s" repeatCount="indefinite" keyTimes="{kt}" values="{cys}"/></circle>')

    d_vals = ";".join(f"M{hp[0]:.0f},{hp[1]:.0f} L{hp[0]-hd[0]*5:.0f},{hp[1]-hd[1]*5:.0f}" for hp, hd in zip(head_pts, head_dirs))
    svg.append(f'<g stroke="{pal["tongue"]}" stroke-width="0.7" fill="none" stroke-linecap="round"><path>'
               f'<animate attributeName="d" dur="{DUR}s" repeatCount="indefinite" keyTimes="{kt}" values="{d_vals}"/>'
               f'<animate attributeName="opacity" dur="0.9s" repeatCount="indefinite" '
               f'values="0;0;1;1;0;0;0;0;0" keyTimes="0;0.5;0.58;0.72;0.8;1;1;1;1"/></path></g>')

    svg.append('</svg>')
    return "".join(svg)

def main():
    token = os.environ["GITHUB_TOKEN"]
    username = os.environ.get("SNAKE_USERNAME") or os.environ["GITHUB_REPOSITORY_OWNER"]
    data = fetch_contributions(username, token)

    os.makedirs("dist", exist_ok=True)
    for mode in ["dark", "light"]:
        svg = build_svg(mode, data)
        with open(f"dist/{mode}.svg", "w") as f:
            f.write(svg)
        print(f"wrote dist/{mode}.svg ({len(svg)/1024:.1f} KB)")

if __name__ == "__main__":
    main()

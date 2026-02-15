#!/usr/bin/env python3
"""
Generate performance report with SVG charts for CI.

Pure Python, zero dependencies — generates SVG charts and a unified markdown
report suitable for GitHub step summaries and PR comments.

Usage:
    python report_charts.py compare baseline.json current.json
    python report_charts.py roofline roofline_report.json
    python report_charts.py full baseline.json current.json    # both
    python report_charts.py trend                              # from runs/
"""

import json
import sys
import os
import math
import argparse
import base64
from html import escape as html_escape


# ─── SVG primitives ────────────────────────────────────────────────────────

COLORS = {
    'baseline':    '#6c757d',   # gray
    'current':     '#0d6efd',   # blue
    'improvement': '#198754',   # green
    'regression':  '#dc3545',   # red
    'neutral':     '#ffc107',   # yellow
    'roofline':    '#fd7e14',   # orange
    'point':       '#0d6efd',   # blue
    'bg':          '#ffffff',
    'grid':        '#e9ecef',
    'text':        '#212529',
    'muted':       '#6c757d',
}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _svg_header(width, height, title=''):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" '
        f'style="font-family: {FONT}; background: {COLORS["bg"]}">\n'
        f'<title>{html_escape(title)}</title>\n'
    )


def _svg_footer():
    return '</svg>\n'


def _nice_ticks(lo, hi, max_ticks=6):
    """Generate human-friendly tick values for an axis."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(max_ticks - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    nice = [1, 2, 2.5, 5, 10]
    step = mag * min(nice, key=lambda n: abs(n * mag - raw))
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.01:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt_num(v):
    """Format a number for axis labels (compact)."""
    if v == 0:
        return '0'
    if abs(v) >= 1000:
        return f'{v:.0f}'
    if abs(v) >= 100:
        return f'{v:.0f}'
    if abs(v) >= 10:
        return f'{v:.1f}'
    if abs(v) >= 1:
        return f'{v:.2f}'
    return f'{v:.3f}'


# ─── Chart generators ──────────────────────────────────────────────────────

def svg_grouped_bar(title, labels, series, y_label='',
                    width=700, height=380, bar_gap=4):
    """Grouped bar chart with multiple series.

    Args:
        title:  chart title
        labels: x-axis category labels
        series: list of {'name': str, 'values': list[float], 'color': str}
        y_label: y-axis label
    Returns:
        SVG string
    """
    n_cats = len(labels)
    n_ser = len(series)
    if n_cats == 0 or n_ser == 0:
        return ''

    # Layout
    margin = {'top': 50, 'right': 30, 'bottom': 70, 'left': 70}
    plot_w = width - margin['left'] - margin['right']
    plot_h = height - margin['top'] - margin['bottom']

    # Data range
    all_vals = [v for s in series for v in s['values'] if v is not None]
    y_max = max(all_vals) if all_vals else 1
    ticks = _nice_ticks(0, y_max)
    y_top = ticks[-1] if ticks else y_max

    # Sizing
    cat_width = plot_w / n_cats
    bar_width = max(8, (cat_width - bar_gap * (n_ser + 1)) / n_ser)

    def x_pos(cat_i, ser_i):
        base = margin['left'] + cat_i * cat_width
        offset = bar_gap + ser_i * (bar_width + bar_gap)
        return base + offset

    def y_pos(val):
        return margin['top'] + plot_h * (1 - val / y_top) if y_top > 0 else margin['top'] + plot_h

    parts = [_svg_header(width, height, title)]

    # Title
    parts.append(
        f'<text x="{width/2}" y="28" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="{COLORS["text"]}">'
        f'{html_escape(title)}</text>\n'
    )

    # Grid lines + y-axis labels
    for t in ticks:
        y = y_pos(t)
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y}" '
            f'x2="{width - margin["right"]}" y2="{y}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        )
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4}" '
            f'text-anchor="end" font-size="11" fill="{COLORS["muted"]}">'
            f'{_fmt_num(t)}</text>\n'
        )

    # Y-axis label
    if y_label:
        yl_x = 15
        yl_y = margin['top'] + plot_h / 2
        parts.append(
            f'<text x="{yl_x}" y="{yl_y}" text-anchor="middle" '
            f'font-size="12" fill="{COLORS["muted"]}" '
            f'transform="rotate(-90 {yl_x} {yl_y})">'
            f'{html_escape(y_label)}</text>\n'
        )

    # Bars
    for ci, label in enumerate(labels):
        for si, s in enumerate(series):
            val = s['values'][ci]
            if val is None:
                continue
            bx = x_pos(ci, si)
            by = y_pos(val)
            bh = y_pos(0) - by
            color = s.get('color', COLORS['current'])
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" '
                f'width="{bar_width:.1f}" height="{max(bh, 0.5):.1f}" '
                f'fill="{color}" rx="2">'
                f'<title>{html_escape(s["name"])}: {_fmt_num(val)}</title>'
                f'</rect>\n'
            )
            # Value label on top of bar
            if bh > 15:
                parts.append(
                    f'<text x="{bx + bar_width/2:.1f}" y="{by - 4:.1f}" '
                    f'text-anchor="middle" font-size="9" fill="{COLORS["muted"]}">'
                    f'{_fmt_num(val)}</text>\n'
                )

        # X-axis label
        cx = margin['left'] + ci * cat_width + cat_width / 2
        cy = margin['top'] + plot_h + 20
        parts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" '
            f'font-size="11" fill="{COLORS["text"]}">'
            f'{html_escape(label)}</text>\n'
        )

    # Legend
    legend_x = margin['left'] + 10
    legend_y = height - 18
    for si, s in enumerate(series):
        lx = legend_x + si * 130
        color = s.get('color', COLORS['current'])
        parts.append(
            f'<rect x="{lx}" y="{legend_y - 9}" width="12" height="12" '
            f'fill="{color}" rx="2"/>\n'
        )
        parts.append(
            f'<text x="{lx + 16}" y="{legend_y + 1}" '
            f'font-size="11" fill="{COLORS["text"]}">'
            f'{html_escape(s["name"])}</text>\n'
        )

    # Axes
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{margin["top"] + plot_h}" '
        f'stroke="{COLORS["muted"]}" stroke-width="1"/>\n'
    )
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" '
        f'x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" '
        f'stroke="{COLORS["muted"]}" stroke-width="1"/>\n'
    )

    parts.append(_svg_footer())
    return ''.join(parts)


def svg_roofline(title, cpu_data, bw_data, results,
                 width=700, height=420):
    """Roofline diagram: arithmetic intensity vs achieved GFLOPS.

    Draws the memory bandwidth ceiling and compute ceiling lines,
    then plots each config as a labeled point.
    """
    peak_gflops = cpu_data.get('peak_gflops', 1)
    dram_bw = bw_data.get('raw_gbs', 1)

    # Ridge point: where memory and compute ceilings meet
    ridge_oi = peak_gflops / dram_bw if dram_bw > 0 else 1

    margin = {'top': 50, 'right': 40, 'bottom': 60, 'left': 70}
    plot_w = width - margin['left'] - margin['right']
    plot_h = height - margin['top'] - margin['bottom']

    # Collect data points
    points = []
    for r in results:
        c = r.get('config', {})
        label = f"e{c.get('n_embd','?')}_L{c.get('n_layer','?')}"
        oi = r.get('oi_c', r.get('oi_python', 0))
        gflops = r.get('achieved_gflops', 0)
        eff = r.get('efficiency_pct', 0)
        if oi > 0 and gflops > 0:
            points.append({'label': label, 'oi': oi, 'gflops': gflops, 'eff': eff})

    if not points:
        return ''

    # Log-log axes
    all_oi = [p['oi'] for p in points] + [ridge_oi]
    all_gf = [p['gflops'] for p in points] + [peak_gflops]

    x_min_log = math.floor(math.log10(min(all_oi)) * 2) / 2
    x_max_log = math.ceil(math.log10(max(all_oi) * 2) * 2) / 2
    y_min_log = math.floor(math.log10(min(all_gf) / 2) * 2) / 2
    y_max_log = math.ceil(math.log10(max(all_gf) * 2) * 2) / 2

    def to_x(oi):
        if oi <= 0:
            return margin['left']
        frac = (math.log10(oi) - x_min_log) / (x_max_log - x_min_log)
        return margin['left'] + frac * plot_w

    def to_y(gf):
        if gf <= 0:
            return margin['top'] + plot_h
        frac = (math.log10(gf) - y_min_log) / (y_max_log - y_min_log)
        return margin['top'] + plot_h * (1 - frac)

    parts = [_svg_header(width, height, title)]

    # Title
    parts.append(
        f'<text x="{width/2}" y="28" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="{COLORS["text"]}">'
        f'{html_escape(title)}</text>\n'
    )

    # Grid lines (log scale)
    for exp in range(int(x_min_log * 2), int(x_max_log * 2) + 1):
        val = 10 ** (exp / 2)
        x = to_x(val)
        if margin['left'] <= x <= margin['left'] + plot_w:
            parts.append(
                f'<line x1="{x:.1f}" y1="{margin["top"]}" '
                f'x2="{x:.1f}" y2="{margin["top"] + plot_h}" '
                f'stroke="{COLORS["grid"]}" stroke-width="0.5"/>\n'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 18}" '
                f'text-anchor="middle" font-size="10" fill="{COLORS["muted"]}">'
                f'{_fmt_num(val)}</text>\n'
            )

    for exp in range(int(y_min_log * 2), int(y_max_log * 2) + 1):
        val = 10 ** (exp / 2)
        y = to_y(val)
        if margin['top'] <= y <= margin['top'] + plot_h:
            parts.append(
                f'<line x1="{margin["left"]}" y1="{y:.1f}" '
                f'x2="{margin["left"] + plot_w}" y2="{y:.1f}" '
                f'stroke="{COLORS["grid"]}" stroke-width="0.5"/>\n'
            )
            parts.append(
                f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" '
                f'text-anchor="end" font-size="10" fill="{COLORS["muted"]}">'
                f'{_fmt_num(val)}</text>\n'
            )

    # Memory bandwidth ceiling (slope = bandwidth in GFLOPS/unit_OI)
    # Attainable GFLOPS = min(peak, OI * BW)
    # On log-log: log(G) = log(OI) + log(BW) → slope 1 line
    mem_oi_start = 10 ** x_min_log
    mem_oi_end = ridge_oi
    mem_gf_start = mem_oi_start * dram_bw
    mem_gf_end = peak_gflops

    # Clip to plot area
    x1, y1 = to_x(mem_oi_start), to_y(mem_gf_start)
    x2, y2 = to_x(mem_oi_end), to_y(mem_gf_end)
    parts.append(
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{COLORS["roofline"]}" stroke-width="2.5" '
        f'stroke-dasharray="6,3"/>\n'
    )

    # Compute ceiling (horizontal at peak_gflops)
    comp_x_start = to_x(ridge_oi)
    comp_x_end = to_x(10 ** x_max_log)
    comp_y = to_y(peak_gflops)
    parts.append(
        f'<line x1="{comp_x_start:.1f}" y1="{comp_y:.1f}" '
        f'x2="{comp_x_end:.1f}" y2="{comp_y:.1f}" '
        f'stroke="{COLORS["regression"]}" stroke-width="2.5" '
        f'stroke-dasharray="6,3"/>\n'
    )

    # Labels for ceilings
    parts.append(
        f'<text x="{x1 + 10:.1f}" y="{y1 - 6:.1f}" '
        f'font-size="10" fill="{COLORS["roofline"]}">'
        f'Memory BW: {dram_bw:.1f} GB/s</text>\n'
    )
    parts.append(
        f'<text x="{comp_x_end - 5:.1f}" y="{comp_y - 6:.1f}" '
        f'text-anchor="end" font-size="10" fill="{COLORS["regression"]}">'
        f'Peak: {peak_gflops:.1f} GFLOPS</text>\n'
    )

    # Data points
    for i, p in enumerate(points):
        px, py = to_x(p['oi']), to_y(p['gflops'])
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" '
            f'fill="{COLORS["point"]}" stroke="white" stroke-width="1.5">'
            f'<title>{html_escape(p["label"])}: OI={p["oi"]:.3f}, '
            f'{p["gflops"]:.4f} GFLOPS, {p["eff"]:.2f}%</title>'
            f'</circle>\n'
        )
        # Label offset to avoid overlap
        label_y_off = -12 if i % 2 == 0 else 18
        parts.append(
            f'<text x="{px:.1f}" y="{py + label_y_off:.1f}" '
            f'text-anchor="middle" font-size="10" font-weight="500" '
            f'fill="{COLORS["text"]}">'
            f'{html_escape(p["label"])}</text>\n'
        )

    # Axis labels
    parts.append(
        f'<text x="{margin["left"] + plot_w/2}" y="{height - 10}" '
        f'text-anchor="middle" font-size="12" fill="{COLORS["muted"]}">'
        f'Arithmetic Intensity (FLOPS/byte)</text>\n'
    )
    yl_x = 15
    yl_y = margin['top'] + plot_h / 2
    parts.append(
        f'<text x="{yl_x}" y="{yl_y}" text-anchor="middle" '
        f'font-size="12" fill="{COLORS["muted"]}" '
        f'transform="rotate(-90 {yl_x} {yl_y})">'
        f'Achieved GFLOPS</text>\n'
    )

    # Axes
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
        f'x2="{margin["left"]}" y2="{margin["top"] + plot_h}" '
        f'stroke="{COLORS["muted"]}" stroke-width="1"/>\n'
    )
    parts.append(
        f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" '
        f'x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" '
        f'stroke="{COLORS["muted"]}" stroke-width="1"/>\n'
    )

    parts.append(_svg_footer())
    return ''.join(parts)


def svg_speedup_bars(title, labels, speedups, width=700, height=320):
    """Horizontal bar chart showing speedup factors with a 1x reference line."""
    n = len(labels)
    if n == 0:
        return ''

    margin = {'top': 45, 'right': 80, 'bottom': 30, 'left': 120}
    plot_w = width - margin['left'] - margin['right']
    plot_h = height - margin['top'] - margin['bottom']
    bar_h = min(30, plot_h / n - 4)

    x_max = max(speedups) * 1.15 if speedups else 2
    ticks = _nice_ticks(0, x_max)
    x_top = ticks[-1] if ticks else x_max

    def to_x(val):
        return margin['left'] + (val / x_top) * plot_w

    parts = [_svg_header(width, height, title)]

    # Title
    parts.append(
        f'<text x="{width/2}" y="26" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="{COLORS["text"]}">'
        f'{html_escape(title)}</text>\n'
    )

    # Grid + x labels
    for t in ticks:
        x = to_x(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{margin["top"]}" '
            f'x2="{x:.1f}" y2="{margin["top"] + plot_h}" '
            f'stroke="{COLORS["grid"]}" stroke-width="1"/>\n'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{margin["top"] + plot_h + 16}" '
            f'text-anchor="middle" font-size="10" fill="{COLORS["muted"]}">'
            f'{_fmt_num(t)}x</text>\n'
        )

    # 1x reference line
    ref_x = to_x(1)
    parts.append(
        f'<line x1="{ref_x:.1f}" y1="{margin["top"]}" '
        f'x2="{ref_x:.1f}" y2="{margin["top"] + plot_h}" '
        f'stroke="{COLORS["muted"]}" stroke-width="1.5" stroke-dasharray="4,3"/>\n'
    )

    # Bars
    for i, (label, spd) in enumerate(zip(labels, speedups)):
        by = margin['top'] + i * (plot_h / n) + (plot_h / n - bar_h) / 2
        bw = to_x(spd) - margin['left']
        color = COLORS['improvement'] if spd >= 1 else COLORS['regression']
        parts.append(
            f'<rect x="{margin["left"]}" y="{by:.1f}" '
            f'width="{max(bw, 1):.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" rx="3" opacity="0.85">'
            f'<title>{html_escape(label)}: {spd:.1f}x</title>'
            f'</rect>\n'
        )
        # Label on left
        parts.append(
            f'<text x="{margin["left"] - 6}" y="{by + bar_h/2 + 4:.1f}" '
            f'text-anchor="end" font-size="11" fill="{COLORS["text"]}">'
            f'{html_escape(label)}</text>\n'
        )
        # Value on right
        parts.append(
            f'<text x="{margin["left"] + bw + 6:.1f}" y="{by + bar_h/2 + 4:.1f}" '
            f'font-size="11" font-weight="600" fill="{color}">'
            f'{spd:.1f}x</text>\n'
        )

    parts.append(_svg_footer())
    return ''.join(parts)


# ─── Report generators ─────────────────────────────────────────────────────

def _config_label(config):
    return f"e{config['n_embd']}_L{config['n_layer']}_b{config['block_size']}"


def _svg_to_img_tag(svg_str, alt='chart'):
    """Convert SVG string to a base64-encoded <img> tag for markdown."""
    b64 = base64.b64encode(svg_str.encode('utf-8')).decode('ascii')
    return f'<img src="data:image/svg+xml;base64,{b64}" alt="{html_escape(alt)}"/>'


def generate_comparison_report(baseline_path, current_path, output_dir='.'):
    """Generate comparison charts and markdown report.

    Returns:
        dict with 'markdown' (str), 'svgs' (dict of name→svg_str)
    """
    with open(baseline_path) as f:
        base_data = json.load(f)
    with open(current_path) as f:
        curr_data = json.load(f)

    base_keyed = {}
    for r in base_data.get('results', []):
        base_keyed[_config_label(r['config'])] = r
    curr_keyed = {}
    for r in curr_data.get('results', []):
        curr_keyed[_config_label(r['config'])] = r

    all_configs = sorted(set(list(base_keyed.keys()) + list(curr_keyed.keys())))

    labels = []
    base_ms = []
    curr_ms = []
    speedups = []
    base_eff = []
    curr_eff = []

    for cfg in all_configs:
        b = base_keyed.get(cfg, {})
        c = curr_keyed.get(cfg, {})
        b_t = b.get('step_time_ms')
        c_t = c.get('step_time_ms')
        if b_t and c_t:
            labels.append(cfg)
            base_ms.append(b_t)
            curr_ms.append(c_t)
            speedups.append(b_t / c_t if c_t > 0 else 0)
            base_eff.append(b.get('efficiency_pct', 0))
            curr_eff.append(c.get('efficiency_pct', 0))

    svgs = {}

    # 1. ms/step comparison bar chart
    svgs['comparison'] = svg_grouped_bar(
        'Step Time Comparison (lower is better)',
        labels,
        [
            {'name': 'Baseline', 'values': base_ms, 'color': COLORS['baseline']},
            {'name': 'Current', 'values': curr_ms, 'color': COLORS['current']},
        ],
        y_label='ms / step',
    )

    # 2. Speedup bar chart
    svgs['speedup'] = svg_speedup_bars(
        'Speedup vs Baseline',
        labels, speedups,
    )

    # 3. Efficiency comparison
    svgs['efficiency'] = svg_grouped_bar(
        'Compute Efficiency (% of peak GFLOPS)',
        labels,
        [
            {'name': 'Baseline', 'values': base_eff, 'color': COLORS['baseline']},
            {'name': 'Current', 'values': curr_eff, 'color': COLORS['current']},
        ],
        y_label='Efficiency %',
        height=340,
    )

    # 4. Roofline diagram (current only)
    svgs['roofline'] = svg_roofline(
        'Roofline Model',
        curr_data.get('cpu', {}),
        curr_data.get('bandwidth', {}),
        curr_data.get('results', []),
    )

    # Save SVGs
    for name, svg in svgs.items():
        path = os.path.join(output_dir, f'chart_{name}.svg')
        with open(path, 'w') as f:
            f.write(svg)

    # Build markdown report
    cpu = curr_data.get('cpu', {})
    md = []
    md.append('<!-- microgpt-benchmark -->')
    md.append('## Performance Report')
    md.append('')
    md.append(f'**CPU:** {cpu.get("model", "?")} ({cpu.get("cores", "?")} cores) | '
              f'**Peak:** {cpu.get("peak_gflops", 0):.1f} GFLOPS')
    md.append('')

    # Summary table
    md.append('### Benchmark Results')
    md.append('')
    md.append('| Config | Baseline | Current | Speedup | Efficiency |')
    md.append('|--------|----------|---------|---------|------------|')
    regressions = 0
    improvements = 0
    for i, cfg in enumerate(labels):
        spd = speedups[i]
        if spd > 1.05:
            icon = ':white_check_mark:'
            improvements += 1
        elif spd < 0.95:
            icon = ':warning:'
            regressions += 1
        else:
            icon = ':heavy_minus_sign:'
        md.append(
            f'| {cfg} | {base_ms[i]:.1f}ms | {curr_ms[i]:.1f}ms | '
            f'{spd:.1f}x {icon} | {curr_eff[i]:.2f}% |'
        )
    md.append('')

    if regressions:
        md.append(f'> :warning: **{regressions} regression(s)** detected')
    if improvements:
        md.append(f'> :white_check_mark: **{improvements} improvement(s)**')
    if not regressions and not improvements:
        md.append('> :heavy_minus_sign: No significant changes')
    md.append('')

    # Embed charts
    md.append('### Step Time Comparison')
    md.append('')
    md.append(_svg_to_img_tag(svgs['comparison'], 'Step time comparison chart'))
    md.append('')

    md.append('### Speedup')
    md.append('')
    md.append(_svg_to_img_tag(svgs['speedup'], 'Speedup chart'))
    md.append('')

    md.append('### Compute Efficiency')
    md.append('')
    md.append(_svg_to_img_tag(svgs['efficiency'], 'Efficiency chart'))
    md.append('')

    md.append('### Roofline Model')
    md.append('')
    md.append(_svg_to_img_tag(svgs['roofline'], 'Roofline diagram'))
    md.append('')

    md.append('_Generated by `report_charts.py` on CI_')

    markdown = '\n'.join(md)

    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(markdown)

    return {'markdown': markdown, 'svgs': svgs}


def generate_roofline_report(roofline_path, output_dir='.'):
    """Generate standalone roofline chart and summary."""
    with open(roofline_path) as f:
        data = json.load(f)

    svg = svg_roofline(
        'Roofline Model',
        data.get('cpu', {}),
        data.get('bandwidth', {}),
        data.get('results', []),
    )

    path = os.path.join(output_dir, 'chart_roofline.svg')
    with open(path, 'w') as f:
        f.write(svg)

    # Efficiency bar chart
    labels = []
    effs = []
    for r in data.get('results', []):
        labels.append(_config_label(r['config']))
        effs.append(r.get('efficiency_pct', 0))

    eff_svg = svg_grouped_bar(
        'Compute Efficiency (% of peak)',
        labels,
        [{'name': 'Efficiency', 'values': effs, 'color': COLORS['current']}],
        y_label='%',
        height=320,
    )
    with open(os.path.join(output_dir, 'chart_efficiency.svg'), 'w') as f:
        f.write(eff_svg)

    return {'roofline_svg': svg, 'efficiency_svg': eff_svg}


def generate_trend_report(output_dir='.'):
    """Generate trend charts from archived benchmark runs."""
    try:
        from run_utils import load_runs
    except ImportError:
        print("run_utils not available for trend report")
        return None

    bench_runs = load_runs(run_type='benchmark')
    if len(bench_runs) < 2:
        print(f"Need >=2 benchmark runs for trend, found {len(bench_runs)}")
        return None

    # Collect timeline data
    configs_seen = {}
    entries = []
    for fname, data in bench_runs:
        meta = data.get('_meta', {})
        git = meta.get('git', {})
        commit = git.get('commit_short', fname[:8])
        row = {'commit': commit, 'configs': {}}
        for r in data.get('results', []):
            label = _config_label(r['config'])
            ms = r.get('step_time_ms')
            if ms is not None:
                row['configs'][label] = ms
                configs_seen[label] = True
        entries.append(row)

    all_configs = sorted(configs_seen.keys())
    commits = [e['commit'] for e in entries]

    # One chart per config showing trend over commits
    svgs = {}
    for cfg in all_configs:
        values = [e['configs'].get(cfg) for e in entries]
        svg = svg_grouped_bar(
            f'Trend: {cfg}',
            commits,
            [{'name': 'ms/step', 'values': values, 'color': COLORS['current']}],
            y_label='ms / step',
            width=max(400, len(commits) * 60 + 150),
            height=280,
        )
        svgs[f'trend_{cfg}'] = svg
        with open(os.path.join(output_dir, f'chart_trend_{cfg}.svg'), 'w') as f:
            f.write(svg)

    return svgs


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate performance charts')
    sub = parser.add_subparsers(dest='command')

    p_cmp = sub.add_parser('compare', help='Compare baseline vs current')
    p_cmp.add_argument('baseline', help='Baseline JSON path')
    p_cmp.add_argument('current', help='Current JSON path')
    p_cmp.add_argument('--output-dir', '-d', default='.', help='Output directory for SVGs')

    p_roof = sub.add_parser('roofline', help='Roofline analysis chart')
    p_roof.add_argument('input', help='Roofline JSON path')
    p_roof.add_argument('--output-dir', '-d', default='.', help='Output directory')

    p_full = sub.add_parser('full', help='Full report (compare + roofline)')
    p_full.add_argument('baseline', help='Baseline JSON path')
    p_full.add_argument('current', help='Current JSON path')
    p_full.add_argument('--output-dir', '-d', default='.', help='Output directory')

    p_trend = sub.add_parser('trend', help='Trend charts from archived runs')
    p_trend.add_argument('--output-dir', '-d', default='.', help='Output directory')

    args = parser.parse_args()

    if args.command == 'compare':
        result = generate_comparison_report(args.baseline, args.current, args.output_dir)
        print(f"Generated {len(result['svgs'])} charts + report.md")

    elif args.command == 'roofline':
        generate_roofline_report(args.input, args.output_dir)
        print("Generated roofline charts")

    elif args.command == 'full':
        result = generate_comparison_report(args.baseline, args.current, args.output_dir)
        generate_roofline_report(args.current, args.output_dir)
        trend = generate_trend_report(args.output_dir)
        n = len(result['svgs']) + 2 + (len(trend) if trend else 0)
        print(f"Generated {n} charts + report.md")

    elif args.command == 'trend':
        trend = generate_trend_report(args.output_dir)
        if trend:
            print(f"Generated {len(trend)} trend charts")
        else:
            print("No trend data available")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

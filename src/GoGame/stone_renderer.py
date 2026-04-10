"""
stone_renderer.py
-----------------
Renders realistic Go stones using per-pixel radial math.
Stones are cached by radius so they're only computed once per session.

Public API:
    draw_stone(surface, color, cx, cy, radius, alpha=255)
        color : 'black' | 'white' | 'territory_black' | 'territory_white'
        alpha : 255 = opaque, 128 = 50% (dead stones)
    clear_cache()
"""

import pygame
import numpy as np
from typing import Tuple

_cache: dict = {}


def clear_cache() -> None:
    _cache.clear()


# ── pixel-level stone rendering ───────────────────────────────────────────────

def _make_stone(radius: int, is_black: bool) -> pygame.Surface:
    """
    Build a stone surface using numpy for per-pixel computation.
    Uses a Phong-like shading model:
      - ambient base colour
      - diffuse shading from a light source upper-left
      - specular highlight (Blinn-Phong) producing the bright crescent
    """
    scale = 2
    r2    = radius * scale
    size  = r2 * 2 + 2
    cx = cy = r2 + 1

    # coordinate grid
    y_idx, x_idx = np.mgrid[0:size, 0:size]
    dx = (x_idx - cx).astype(np.float32)
    dy = (y_idx - cy).astype(np.float32)
    dist = np.sqrt(dx * dx + dy * dy)

    # mask: pixels inside the stone
    mask = dist <= radius

    # normalised surface normal (sphere)
    # nx, ny from (dx,dy)/r, nz = sqrt(1 - nx²- ny²)
    with np.errstate(invalid='ignore', divide='ignore'):
        nx = np.where(mask, dx / radius, 0.0)
        ny = np.where(mask, dy / radius, 0.0)
        nz_sq = np.clip(1.0 - nx*nx - ny*ny, 0, 1)
        nz = np.sqrt(nz_sq)

    # light direction — upper left, slightly in front
    lx, ly, lz = -0.55, -0.65, 0.52
    ln = np.sqrt(lx*lx + ly*ly + lz*lz)
    lx, ly, lz = lx/ln, ly/ln, lz/ln

    # diffuse term (Lambert)
    diffuse = np.clip(nx*lx + ny*ly + nz*lz, 0, 1)

    # specular term (Blinn-Phong)
    # half-vector between light and view (view = (0,0,1))
    hx, hy, hz = lx, ly, lz + 1.0
    hn = np.sqrt(hx*hx + hy*hy + hz*hz)
    hx, hy, hz = hx/hn, hy/hn, hz/hn
    spec_dot = np.clip(nx*hx + ny*hy + nz*hz, 0, 1)

    if is_black:
        shininess  = 60
        spec_power = np.power(spec_dot, shininess)
        # base: very dark, slight blue-grey tint
        amb = np.array([0.06, 0.06, 0.07], dtype=np.float32)
        dif = np.array([0.18, 0.18, 0.20], dtype=np.float32)
        spec_col = np.array([0.95, 0.97, 1.00], dtype=np.float32)
        spec_str = 1.1
    else:
        shininess  = 45
        spec_power = np.power(spec_dot, shininess)
        # base: warm creamy white
        amb = np.array([0.72, 0.70, 0.65], dtype=np.float32)
        dif = np.array([0.55, 0.53, 0.48], dtype=np.float32)
        spec_col = np.array([1.00, 1.00, 1.00], dtype=np.float32)
        spec_str = 0.85

    # combine: ambient + diffuse*light + specular
    r_ch = amb[0] + dif[0]*diffuse + spec_col[0]*spec_power*spec_str
    g_ch = amb[1] + dif[1]*diffuse + spec_col[1]*spec_power*spec_str
    b_ch = amb[2] + dif[2]*diffuse + spec_col[2]*spec_power*spec_str

    r_ch = np.clip(r_ch, 0, 1)
    g_ch = np.clip(g_ch, 0, 1)
    b_ch = np.clip(b_ch, 0, 1)

    # soft edge anti-alias — fade alpha over outer 1.5px
    edge_dist = radius - dist
    fade_width = max(1.5, radius * 0.12)
    alpha_ch  = np.clip(edge_dist / fade_width + 1.0, 0, 1)
    alpha_ch  = np.where(mask, alpha_ch, 0.0)

    # pack into RGBA array
    pixels = np.zeros((size, size, 4), dtype=np.uint8)
    pixels[:, :, 0] = (r_ch * 255).astype(np.uint8)
    pixels[:, :, 1] = (g_ch * 255).astype(np.uint8)
    pixels[:, :, 2] = (b_ch * 255).astype(np.uint8)
    pixels[:, :, 3] = (alpha_ch * 255).astype(np.uint8)

    # zero out pixels outside mask entirely
    pixels[:, :, 3] = np.where(mask, pixels[:, :, 3], 0)

    buf  = np.ascontiguousarray(pixels)
    big  = pygame.image.frombuffer(buf.tobytes(), (size, size), 'RGBA')
    final_size = radius * 2 + 2
    surf = pygame.transform.smoothscale(big, (final_size, final_size))
    return surf.convert_alpha()


def _make_territory_marker(radius: int, is_black: bool) -> pygame.Surface:
    """
    Small rounded square territory marker.
    Black territory = dark square, white territory = light square.
    Semi-transparent so board color shows through.
    """
    size = radius * 2 + 2
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = radius + 1
    mr = max(3, int(radius * 0.42))

    if is_black:
        fill   = (15,  15,  18,  210)
        border = (5,   5,   8,  240)
    else:
        fill   = (238, 235, 228, 210)
        border = (180, 175, 165, 240)

    rect = pygame.Rect(cx - mr, cy - mr, mr * 2, mr * 2)
    pygame.draw.rect(surf, fill,   rect, border_radius=3)
    pygame.draw.rect(surf, border, rect, width=1, border_radius=3)
    return surf


def _get_surface(color: str, radius: int) -> pygame.Surface:
    key = (color, radius)
    if key not in _cache:
        if color == 'black':
            _cache[key] = _make_stone(radius, is_black=True)
        elif color == 'white':
            _cache[key] = _make_stone(radius, is_black=False)
        elif color == 'territory_black':
            _cache[key] = _make_territory_marker(radius, is_black=True)
        elif color == 'territory_white':
            _cache[key] = _make_territory_marker(radius, is_black=False)
    return _cache[key]


# ── public API ────────────────────────────────────────────────────────────────

def draw_stone(surface: pygame.Surface, color: str,
               cx: int, cy: int, radius: int, alpha: int = 255) -> None:
    """
    Draw a stone or territory marker onto *surface* centred at (cx, cy).

    Parameters:
        surface : destination pygame.Surface
        color   : 'black' | 'white' | 'territory_black' | 'territory_white'
        cx, cy  : centre pixel coordinates
        radius  : stone radius in pixels
        alpha   : 255 = opaque, 128 = 50% transparent (dead stones)
    """
    base = _get_surface(color, radius)
    x    = cx - radius - 1
    y    = cy - radius - 1

    if alpha >= 255:
        surface.blit(base, (x, y))
    else:
        faded = base.copy()
        faded.set_alpha(alpha)
        surface.blit(faded, (x, y))

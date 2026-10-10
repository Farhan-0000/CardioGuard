"""
Tiny matplotlib layout kit for hand-placed diagrams (inches, y grows downwards).

Used by the architecture and sequence diagrams, where explicit placement gives a
cleaner, more compact result than automatic Graphviz layout. ``Canvas.check()``
measures every rendered text against its box and raises if a text overflows,
so layout regressions are caught when the figures are rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from _common import EDGE, PALETTE, SUBTEXT, setup_matplotlib

setup_matplotlib()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

DASH = (0, (4.0, 2.4))
DOT = (0, (1.2, 1.8))


@dataclass
class Box:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.cx - self.w / 2

    @property
    def right(self) -> float:
        return self.cx + self.w / 2

    @property
    def top(self) -> float:
        return self.cy - self.h / 2

    @property
    def bottom(self) -> float:
        return self.cy + self.h / 2


@dataclass
class Canvas:
    width: float
    height: float
    texts: list[tuple[object, Box, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fig = plt.figure(figsize=(self.width, self.height))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

    def resize(self, height: float) -> None:
        """Change the canvas height (content keeps its inch coordinates)."""
        self.height = height
        self.fig.set_size_inches(self.width, height)
        self.ax.set_ylim(height, 0)

    def trim(self, top: float = 0.0, bottom: float | None = None) -> None:
        """Show only the band ``top``..``bottom`` (inches) — removes empty margins."""
        bottom = self.height if bottom is None else bottom
        self.fig.set_size_inches(self.width, bottom - top)
        self.ax.set_ylim(bottom, top)

    # -- primitives ---------------------------------------------------------

    def _text_block(
        self,
        box: Box,
        title: str,
        lines: tuple[str, ...] | list[str],
        kind: str,
        title_size: float,
        sub_size: float,
        z: float,
        name: str,
    ) -> None:
        k = PALETTE[kind]
        th = title_size / 72 * 1.22
        lh = sub_size / 72 * 1.22
        total = (th if title else 0) + lh * len(lines)
        y = box.cy - total / 2
        if title:
            t = self.ax.text(
                box.cx,
                y + th / 2,
                title,
                ha="center",
                va="center",
                fontsize=title_size,
                fontweight="bold",
                color=k.text,
                zorder=z,
            )
            self.texts.append((t, box, name))
            y += th
        for line in lines:
            t = self.ax.text(
                box.cx,
                y + lh / 2,
                line,
                ha="center",
                va="center",
                fontsize=sub_size,
                color=SUBTEXT,
                zorder=z,
            )
            self.texts.append((t, box, name))
            y += lh

    def box(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        title: str,
        lines: tuple[str, ...] | list[str] = (),
        *,
        kind: str = "det",
        optional: bool = False,
        title_size: float = 8.5,
        sub_size: float = 7.0,
        lw: float = 1.2,
        rounding: float = 0.07,
    ) -> Box:
        k = PALETTE[kind]
        b = Box(cx, cy, w, h)
        patch = FancyBboxPatch(
            (b.left, b.top),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=k.fill,
            edgecolor=k.line,
            linewidth=lw,
            linestyle=DASH if optional else "solid",
            zorder=3,
        )
        self.ax.add_patch(patch)
        self._text_block(b, title, lines, kind, title_size, sub_size, 4, title or lines[0])
        return b

    def header(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        lines: list[str],
        *,
        kind: str = "det",
        size: float = 7.2,
        tag: str | None = None,
        tag_size: float = 6.2,
    ) -> Box:
        """Box whose lines are all bold (lifeline heads), plus an optional small tag line."""
        k = PALETTE[kind]
        b = Box(cx, cy, w, h)
        self.ax.add_patch(
            FancyBboxPatch(
                (b.left, b.top),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=k.fill,
                edgecolor=k.line,
                linewidth=1.2,
                zorder=3,
            )
        )
        lh = size / 72 * 1.2
        th = tag_size / 72 * 1.2 if tag else 0.0
        y = cy - (lh * len(lines) + th) / 2
        for line in lines:
            t = self.ax.text(
                cx,
                y + lh / 2,
                line,
                ha="center",
                va="center",
                fontsize=size,
                fontweight="bold",
                color=k.text,
                zorder=4,
            )
            self.texts.append((t, b, lines[0]))
            y += lh
        if tag:
            t = self.ax.text(
                cx,
                y + th / 2,
                tag,
                ha="center",
                va="center",
                fontsize=tag_size,
                color=SUBTEXT,
                zorder=4,
            )
            self.texts.append((t, b, lines[0]))
        return b

    def cylinder(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        title: str,
        lines: tuple[str, ...] | list[str] = (),
        *,
        kind: str = "storage",
        title_size: float = 8.5,
        sub_size: float = 7.0,
        lw: float = 1.2,
    ) -> Box:
        k = PALETTE[kind]
        b = Box(cx, cy, w, h)
        cap = min(0.12, h * 0.22)
        self.ax.add_patch(
            Rectangle(
                (b.left, b.top + cap / 2), w, h - cap, facecolor=k.fill, edgecolor="none", zorder=3
            )
        )
        self.ax.plot(
            [b.left, b.left], [b.top + cap / 2, b.bottom - cap / 2], color=k.line, lw=lw, zorder=3.1
        )
        self.ax.plot(
            [b.right, b.right],
            [b.top + cap / 2, b.bottom - cap / 2],
            color=k.line,
            lw=lw,
            zorder=3.1,
        )
        self.ax.add_patch(
            Ellipse(
                (cx, b.bottom - cap / 2),
                w,
                cap,
                facecolor=k.fill,
                edgecolor=k.line,
                lw=lw,
                zorder=2.9,
            )
        )
        self.ax.add_patch(
            Ellipse(
                (cx, b.top + cap / 2), w, cap, facecolor=k.fill, edgecolor=k.line, lw=lw, zorder=3.2
            )
        )
        inner = Box(cx, cy + cap / 3, w, h - cap)
        self._text_block(inner, title, lines, kind, title_size, sub_size, 4, title)
        return b

    def store(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        sid: str,
        title: str,
        lines: tuple[str, ...] | list[str] = (),
        *,
        kind: str = "storage",
        title_size: float = 8.0,
        sub_size: float = 6.6,
        lw: float = 1.3,
        id_w: float = 0.36,
    ) -> Box:
        """DFD data store: open-ended rectangle with an identifier cell on the left."""
        k = PALETTE[kind]
        b = Box(cx, cy, w, h)
        self.ax.add_patch(
            Rectangle((b.left, b.top), w, h, facecolor=k.fill, edgecolor="none", zorder=3)
        )
        self.ax.plot(
            [b.right, b.left, b.left, b.right],
            [b.top, b.top, b.bottom, b.bottom],
            color=k.line,
            lw=lw,
            zorder=3.1,
            solid_joinstyle="miter",
        )
        self.ax.plot([b.left + id_w] * 2, [b.top, b.bottom], color=k.line, lw=lw * 0.8, zorder=3.1)
        t = self.ax.text(
            b.left + id_w / 2,
            cy,
            sid,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=k.text,
            zorder=4,
        )
        self.texts.append((t, Box(b.left + id_w / 2, cy, id_w, h), sid))
        inner = Box(b.left + id_w + (w - id_w) / 2, cy, w - id_w - 0.04, h)
        self._text_block(inner, title, lines, kind, title_size, sub_size, 4, title)
        return b

    def diamond(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        title: str,
        lines: tuple[str, ...] | list[str] = (),
        *,
        kind: str = "det",
        title_size: float = 8.0,
        sub_size: float = 7.0,
        lw: float = 1.2,
    ) -> Box:
        k = PALETTE[kind]
        b = Box(cx, cy, w, h)
        pts = [(cx, b.top), (b.right, cy), (cx, b.bottom), (b.left, cy)]
        self.ax.add_patch(
            Polygon(pts, closed=True, facecolor=k.fill, edgecolor=k.line, lw=lw, zorder=3)
        )
        # text must stay inside the inscribed rectangle (half width / half height)
        self._text_block(
            Box(cx, cy, w / 2, h / 2), title, lines, kind, title_size, sub_size, 4, title
        )
        return b

    def group(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        title: str,
        *,
        kind: str = "det",
        title_size: float = 8.5,
        title_ha: str = "left",
    ) -> Box:
        k = PALETTE[kind]
        self.ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="round,pad=0,rounding_size=0.09",
                facecolor="#F7F8FA",
                edgecolor=k.line,
                lw=1.0,
                linestyle=DOT,
                zorder=1,
            )
        )
        tx = x0 + 0.09 if title_ha == "left" else x1 - 0.09
        self.ax.text(
            tx,
            y0 + 0.06,
            title,
            ha=title_ha,
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=k.text,
            zorder=2,
        )
        return Box((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)

    def arrow(
        self,
        pts: list[tuple[float, float]],
        *,
        color: str = EDGE,
        dashed: bool = False,
        lw: float = 1.1,
        head: bool = True,
        z: float = 2.5,
        both: bool = False,
    ) -> None:
        ls = DASH if dashed else "solid"
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if both and len(pts) >= 2:
            self.ax.annotate(
                "",
                xy=pts[0],
                xytext=pts[1],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    linestyle=ls,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=8.5,
                ),
                zorder=z,
            )
        if head and len(pts) >= 2:
            self.ax.plot(
                xs[:-1],
                ys[:-1],
                color=color,
                lw=lw,
                linestyle=ls,
                zorder=z,
                solid_capstyle="butt",
                dash_capstyle="butt",
            )
            self.ax.annotate(
                "",
                xy=pts[-1],
                xytext=pts[-2],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=lw,
                    linestyle=ls,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=8.5,
                ),
                zorder=z,
            )
        else:
            self.ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=z)

    def junction(self, x: float, y: float, color: str = EDGE) -> None:
        self.ax.plot([x], [y], marker="o", ms=3.2, color=color, zorder=3)

    def label(
        self,
        x: float,
        y: float,
        text: str,
        *,
        color: str = EDGE,
        size: float = 6.8,
        ha: str = "center",
        va: str = "center",
        bg: bool = True,
        bold: bool = False,
        italic: bool = False,
        rotation: float = 0,
        z: float = 5,
    ) -> object:
        bbox = dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none") if bg else None
        return self.ax.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=size,
            color=color,
            bbox=bbox,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            rotation=rotation,
            zorder=z,
            linespacing=1.15,
        )

    def legend(
        self,
        y: float,
        items: list[tuple[str, str]],
        *,
        x0: float = 0.2,
        dashed_note: str | None = None,
        per_row: int = 4,
        col_w: float | None = None,
        size: float = 7.0,
    ) -> None:
        col_w = col_w or (self.width - 2 * x0) / per_row
        for i, (kind, text) in enumerate(items):
            row, col = divmod(i, per_row)
            x = x0 + col * col_w
            yy = y + row * 0.2
            k = PALETTE[kind]
            self.ax.add_patch(
                FancyBboxPatch(
                    (x, yy - 0.05),
                    0.2,
                    0.1,
                    boxstyle="round,pad=0,rounding_size=0.02",
                    facecolor=k.fill,
                    edgecolor=k.line,
                    lw=1.0,
                    zorder=3,
                )
            )
            self.ax.text(x + 0.27, yy, text, ha="left", va="center", fontsize=size, color="#1F2933")
        if dashed_note:
            rows = (len(items) + per_row - 1) // per_row
            self.ax.text(
                x0,
                y + rows * 0.2,
                dashed_note,
                ha="left",
                va="center",
                fontsize=size,
                color=SUBTEXT,
            )

    # -- quality gate -------------------------------------------------------

    def check(self, tolerance: float = 0.02) -> None:
        """Raise if any box text overflows its box (measured on the real renderer)."""
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()  # type: ignore[attr-defined]
        inv = self.ax.transData.inverted()
        problems = []
        for t, b, name in self.texts:
            bb = t.get_window_extent(renderer=renderer)  # type: ignore[attr-defined]
            (x0, y1), (x1, y0) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            if x0 < b.left - tolerance or x1 > b.right + tolerance:
                problems.append(f"{name!r}: text {x1 - x0:.2f} in wide, box {b.w:.2f} in")
            if y0 < b.top - tolerance or y1 > b.bottom + tolerance:
                problems.append(f"{name!r}: text exceeds box height ({b.h:.2f} in)")
        if problems:
            raise ValueError("Text overflow:\n  " + "\n  ".join(problems))

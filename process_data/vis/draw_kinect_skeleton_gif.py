import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


AZURE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 26),
    (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (7, 9), (7, 10),
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (14, 16), (14, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
    (0, 22), (22, 23), (23, 24), (24, 25),
    (3, 27), (27, 28), (28, 29), (27, 30), (30, 31),
]


def load_skeleton(path):
    skeleton = np.load(path)
    if skeleton.ndim != 3 or skeleton.shape[1:] != (32, 3):
        raise ValueError(f"Expected skeleton shape (frames, 32, 3), got {skeleton.shape}: {path}")
    if not np.isfinite(skeleton).all():
        raise ValueError(f"Skeleton contains NaN or Inf values: {path}")
    return skeleton.astype(float)


def skeleton_to_front_view_m(skeleton_mm):
    """Return x lateral and upright height in metres.

    Azure Kinect body coordinates use a downward-positive vertical axis for the
    saved skeletons in this dataset. We flip y so the head is drawn above the
    torso. This same convention also works after rigid Kinect-to-radar
    transformation, because the regenerated aligned skeleton still has y as the
    vertical coordinate.
    """
    x = skeleton_mm[..., 0] / 1000.0
    height = -skeleton_mm[..., 1] / 1000.0
    return np.stack([x, height], axis=-1)


def compute_view_bounds(*point_sets, padding=0.18):
    points = np.concatenate([pts.reshape(-1, 2) for pts in point_sets], axis=0)
    lo = np.nanpercentile(points, 1, axis=0)
    hi = np.nanpercentile(points, 99, axis=0)
    span = np.maximum(hi - lo, np.array([0.8, 1.6]))
    center = (lo + hi) / 2.0
    half = span * (0.5 + padding)
    return center - half, center + half


def project(points_m, lo, hi, width, height, margin):
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    span = np.maximum(hi - lo, 1e-9)
    x = margin + (points_m[:, 0] - lo[0]) / span[0] * plot_w
    y = height - margin - (points_m[:, 1] - lo[1]) / span[1] * plot_h
    return np.stack([x, y], axis=-1)


def get_font(size=18):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_frame(points_m, frame_index, total_frames, lo, hi, title, width, height, margin):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = get_font(18)
    small_font = get_font(14)

    # Light grid.
    grid_color = (230, 230, 230)
    for i in range(6):
        x = margin + i * (width - 2 * margin) / 5
        y = margin + i * (height - 2 * margin) / 5
        draw.line([(x, margin), (x, height - margin)], fill=grid_color, width=1)
        draw.line([(margin, y), (width - margin, y)], fill=grid_color, width=1)

    draw.rectangle([margin, margin, width - margin, height - margin], outline=(90, 90, 90), width=2)
    draw.text((margin, 16), title, fill=(20, 20, 20), font=font)
    draw.text(
        (margin, height - margin + 14),
        f"frame {frame_index + 1}/{total_frames} | x lateral vs upright height",
        fill=(70, 70, 70),
        font=small_font,
    )

    projected = project(points_m, lo, hi, width, height, margin)
    bone_color = (0, 114, 178)
    joint_color = (213, 94, 0)
    head_color = (0, 130, 85)

    for start, end in AZURE_BONES:
        p0 = tuple(projected[start])
        p1 = tuple(projected[end])
        draw.line([p0, p1], fill=bone_color, width=5)

    radius = 5
    for joint_id, point in enumerate(projected):
        color = head_color if joint_id == 26 else joint_color
        x, y = point
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline="white", width=1)

    return image


def save_skeleton_gif(skeleton_mm, out_path, title, lo, hi, every=1, duration_ms=55, width=720, height=720):
    points = skeleton_to_front_view_m(skeleton_mm)
    frame_indices = list(range(0, len(points), every))
    frames = [
        draw_frame(points[idx], idx, len(points), lo, hi, title, width, height, margin=70)
        for idx in frame_indices
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms * every,
        loop=0,
        optimize=True,
    )
    return {
        "path": str(out_path),
        "source_frames": int(len(points)),
        "rendered_frames": int(len(frames)),
        "frame_step": int(every),
        "duration_ms_per_rendered_frame": int(duration_ms * every),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render before/after Kinect skeleton GIFs for one action segment."
    )
    parser.add_argument(
        "--raw-skeleton",
        default="dataset/dataset/env1/subjects/subject26/original/1/skeleton_segments/action01.npy",
        help="Raw/pre-alignment Kinect skeleton segment (.npy).",
    )
    parser.add_argument(
        "--aligned-skeleton",
        default="dataset/dataset/env1/subjects/subject26/aligned/action01/aligned_skeleton_segment01.npy",
        help="Aligned/post-transformation skeleton segment (.npy).",
    )
    parser.add_argument(
        "--out-dir",
        default="process_data/validation/kinect_skeleton_gifs",
        help="Directory for generated GIFs and metadata.",
    )
    parser.add_argument("--every", type=int, default=1, help="Render every Nth frame.")
    parser.add_argument("--duration-ms", type=int, default=55, help="GIF frame duration before frame skipping.")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    raw_path = Path(args.raw_skeleton)
    aligned_path = Path(args.aligned_skeleton)
    out_dir = Path(args.out_dir)

    raw = load_skeleton(raw_path)
    aligned = load_skeleton(aligned_path)
    raw_points = skeleton_to_front_view_m(raw)
    aligned_points = skeleton_to_front_view_m(aligned)
    lo, hi = compute_view_bounds(raw_points, aligned_points)

    raw_meta = save_skeleton_gif(
        raw,
        out_dir / "env1_subject26_action01_kinect_before_alignment.gif",
        "env1 subject26 action01 - raw Kinect skeleton",
        lo,
        hi,
        every=max(args.every, 1),
        duration_ms=args.duration_ms,
        width=args.width,
        height=args.height,
    )
    aligned_meta = save_skeleton_gif(
        aligned,
        out_dir / "env1_subject26_action01_kinect_after_alignment.gif",
        "env1 subject26 action01 - aligned Kinect skeleton",
        lo,
        hi,
        every=max(args.every, 1),
        duration_ms=args.duration_ms,
        width=args.width,
        height=args.height,
    )

    metadata = {
        "raw_skeleton": str(raw_path),
        "aligned_skeleton": str(aligned_path),
        "display_mapping": "x = skeleton_x / 1000, upright_height = -skeleton_y / 1000",
        "view_bounds_m": {"min": lo.tolist(), "max": hi.tolist()},
        "raw_gif": raw_meta,
        "aligned_gif": aligned_meta,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "env1_subject26_action01_kinect_gif_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

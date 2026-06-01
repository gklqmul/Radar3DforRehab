import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot


AZURE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 26),
    (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),
    (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),
    (0, 18), (18, 19), (19, 20), (20, 21),
    (0, 22), (22, 23), (23, 24), (24, 25)
]


JOINT_NAMES = [
    "pelvis", "spine_navel", "spine_chest", "neck",
    "clavicle_l", "shoulder_l", "elbow_l", "wrist_l", "hand_l",
    "handtip_l", "thumb_l", "clavicle_r", "shoulder_r", "elbow_r",
    "wrist_r", "hand_r", "handtip_r", "thumb_r", "hip_l", "knee_l",
    "ankle_l", "foot_l", "hip_r", "knee_r", "ankle_r", "foot_r",
    "head", "nose", "eye_l", "ear_l", "eye_r", "ear_r",
]


def parse_timestamp(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    matches = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", str(value))
    return float(matches[0]) if matches else np.nan


def read_h5_frame(h5_path, frame_index):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["frames"].keys())
        frame_index = min(max(frame_index, 0), len(keys) - 1)
        dataset = f["frames"][keys[frame_index]]
        frame = np.asarray(dataset)
        timestamp = parse_timestamp(dataset.attrs.get("timestamp", np.nan))
    return frame, timestamp, frame_index, len(keys)


def read_h5_frame_stack(h5_path, frame_index, stack_radius):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["frames"].keys())
        frame_index = min(max(frame_index, 0), len(keys) - 1)
        selected = []
        selected_indices = []
        for idx in range(frame_index - stack_radius, frame_index + stack_radius + 1):
            idx = min(max(idx, 0), len(keys) - 1)
            selected_indices.append(idx)
            selected.append(np.asarray(f["frames"][keys[idx]]))
        dataset = f["frames"][keys[frame_index]]
        timestamp = parse_timestamp(dataset.attrs.get("timestamp", np.nan))
    return selected, selected_indices, timestamp, frame_index, len(keys)


def radar_xyz(frame):
    if frame.ndim != 2 or frame.shape[1] < 7:
        return np.zeros((0, 3), dtype=float)
    # Human-readable radar axes: x lateral, y range, z height.
    xyz = np.column_stack([frame[:, 5], frame[:, 6], frame[:, 1]]).astype(float)
    return xyz[np.isfinite(xyz).all(axis=1)]


def skeleton_xyz(skeleton_mm):
    # The regenerated aligned skeleton stores x as lateral, y as height, and
    # negative z as forward range. The height axis is also sign-flipped for
    # display so the head appears above the torso.
    return np.column_stack([
        skeleton_mm[:, 0] / 1000.0,
        -skeleton_mm[:, 2] / 1000.0,
        -skeleton_mm[:, 1] / 1000.0,
    ])


def axis_ranges(radar_points, skeleton_points):
    if len(radar_points):
        points = np.vstack([radar_points, skeleton_points])
    else:
        points = skeleton_points
    lo = np.nanpercentile(points, 1, axis=0)
    hi = np.nanpercentile(points, 99, axis=0)
    center = (lo + hi) / 2
    span = np.maximum(hi - lo, np.array([1.2, 1.2, 1.2]))
    radius = float(np.max(span) / 2)
    return [
        [float(center[0] - radius), float(center[0] + radius)],
        [float(center[1] - radius), float(center[1] + radius)],
        [float(center[2] - radius), float(center[2] + radius)],
    ]


def make_figure(action_dir, frame_index, stack_radius=0):
    action_dir = Path(action_dir)
    h5_path = next(action_dir.glob("*.h5"))
    skeleton_path = next(action_dir.glob("*.npy"))

    radar_frames, stack_indices, timestamp, frame_index, total_frames = read_h5_frame_stack(
        h5_path, frame_index, stack_radius
    )
    skeleton = np.load(skeleton_path)
    skeleton_points = skeleton_xyz(skeleton[frame_index])
    radar_chunks = [radar_xyz(frame) for frame in radar_frames]
    radar_points = (
        np.vstack([chunk for chunk in radar_chunks if len(chunk)])
        if any(len(chunk) for chunk in radar_chunks)
        else np.zeros((0, 3), dtype=float)
    )

    traces = []
    if len(radar_points):
        traces.append(
            go.Scatter3d(
                x=radar_points[:, 0],
                y=radar_points[:, 1],
                z=radar_points[:, 2],
                mode="markers",
                name=f"Radar points (n={len(radar_points)}, stack={2 * stack_radius + 1})",
                marker=dict(size=4, color="#f28e2b", opacity=0.78),
                hovertemplate="radar<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>",
            )
        )

    traces.append(
        go.Scatter3d(
            x=skeleton_points[:, 0],
            y=skeleton_points[:, 1],
            z=skeleton_points[:, 2],
            mode="markers+text",
            name="Skeleton joints",
            marker=dict(size=5, color="#1f77b4", opacity=0.95),
            text=[str(i) for i in range(len(skeleton_points))],
            textposition="top center",
            hovertext=JOINT_NAMES,
            hovertemplate="joint %{text}: %{hovertext}<br>x=%{x:.3f} m<br>y=%{y:.3f} m<br>z=%{z:.3f} m<extra></extra>",
        )
    )

    for bone_id, (start, end) in enumerate(AZURE_BONES):
        traces.append(
            go.Scatter3d(
                x=[skeleton_points[start, 0], skeleton_points[end, 0]],
                y=[skeleton_points[start, 1], skeleton_points[end, 1]],
                z=[skeleton_points[start, 2], skeleton_points[end, 2]],
                mode="lines",
                name="Skeleton bones" if bone_id == 0 else None,
                showlegend=bone_id == 0,
                line=dict(color="#1f77b4", width=6),
                hoverinfo="skip",
            )
        )

    ranges = axis_ranges(radar_points, skeleton_points)
    title = (
        f"{action_dir.as_posix()}<br>"
        f"frame {frame_index}/{total_frames - 1}, timestamp={timestamp:.3f}, "
        f"radar points={len(radar_points)}, stack radius={stack_radius}"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        width=1150,
        height=850,
        scene=dict(
            xaxis=dict(title="x lateral (m)", range=ranges[0], backgroundcolor="rgb(245,245,245)"),
            yaxis=dict(title="y range (m)", range=ranges[1], backgroundcolor="rgb(245,245,245)"),
            zaxis=dict(title="z height (m)", range=ranges[2], backgroundcolor="rgb(245,245,245)"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.1)),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    metadata = {
        "action_dir": str(action_dir),
        "h5_file": str(h5_path),
        "skeleton_file": str(skeleton_path),
        "frame_index": int(frame_index),
        "stack_radius": int(stack_radius),
        "stack_frame_indices": [int(idx) for idx in stack_indices],
        "num_frames": int(total_frames),
        "timestamp": float(timestamp),
        "num_radar_points": int(len(radar_points)),
    }
    return fig, metadata


def write_html(action_dir, frame_index, out_path, stack_radius=0):
    fig, metadata = make_figure(action_dir, frame_index, stack_radius=stack_radius)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(fig, filename=str(out_path), auto_open=False, include_plotlyjs=True)
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Create interactive radar/skeleton overlay HTML.")
    parser.add_argument("--action-dir", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--stack-radius", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata = write_html(args.action_dir, args.frame, args.out, stack_radius=args.stack_radius)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

#  python process_data/validation/draw_radar_Kinect.py \
#   --action-dir dataset/dataset/env1/subjects/subject26/aligned/action16 \
#   --frame 78 \
#   --stack-radius 0 \
#   --out process_data/validation/example_action16_frame078.html
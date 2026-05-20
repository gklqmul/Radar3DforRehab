import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_data.class_files.action_segmenter import ActionSegmenter


ACADEMIC_COLORS = {
    "skeleton": "#0072B2",
    "radar": "#D55E00",
}


def find_action_dir(dataset_root, env, subject, action):
    return Path(dataset_root) / env / "subjects" / subject / "aligned" / action


def parse_timestamp(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    matches = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", str(value))
    return float(matches[0]) if matches else np.nan


def read_h5(path):
    import h5py

    frames = []
    timestamps = []
    with h5py.File(path, "r") as f:
        for key in sorted(f["frames"].keys()):
            dataset = f["frames"][key]
            frames.append(np.asarray(dataset))
            timestamps.append(parse_timestamp(dataset.attrs.get("timestamp", np.nan)))
    return frames, np.asarray(timestamps, dtype=float)


def load_aligned_action(action_dir):
    h5_path = next(action_dir.glob("*.h5"))
    skeleton_path = next(action_dir.glob("*.npy"))
    radar_frames, _ = read_h5(h5_path)
    skeleton = np.load(skeleton_path)
    return radar_frames, skeleton


def compute_skeleton_motion_energy(skeleton_frames, smooth_sigma):
    segmenter = ActionSegmenter(frame_count=len(skeleton_frames), smooth_sigma=smooth_sigma)
    smoothed = segmenter._smooth_frames(skeleton_frames)
    return segmenter._calculate_motion_energy(smoothed)


import numpy as np

def compute_radar_motion_energy(radar_frames, window_size=11):
    """
    计算雷达运动能量，并通过滑动平均消除高频毛刺，恢复动作的宏观周期性。
    
    参数:
    - radar_frames: 雷达帧列表
    - window_size: 平滑窗口大小（奇数，如 5, 7, 9）。数值越大曲线越平滑，波峰越明显。
    """
    frame_abs_doppler_sum = []
    for frame in radar_frames:
        if frame.ndim != 2 or frame.shape[1] <= 3 or frame.shape[0] == 0:
            frame_abs_doppler_sum.append(0.0)
            continue
        doppler = np.asarray(frame[:, 3], dtype=float)
        doppler = doppler[np.isfinite(doppler)]
        # 1. 先取绝对值再求和，防止正负多普勒抵消（你已经做到了这一步，非常棒）
        frame_abs_doppler_sum.append(float(np.sum(np.abs(doppler))))

    frame_abs_doppler_sum = np.asarray(frame_abs_doppler_sum, dtype=float)
    motion_energy = np.zeros_like(frame_abs_doppler_sum)
    
    # 2. 计算帧间多普勒能量相差值
    if len(frame_abs_doppler_sum) > 1:
        motion_energy[1:] = np.abs(np.diff(frame_abs_doppler_sum))
    
    # 3. 核心改进：引入滑动平均（Moving Average）
    # 消除 np.diff 带来的高频锯齿噪声，将断裂的动作连贯起来
    if len(motion_energy) > window_size:
        # 构造平滑滤波器核（所有权重相等且和为 1）
        kernel = np.ones(window_size) / window_size
        # mode='same' 确保平滑后的数组长度与原始数组完全一致，方便后续对齐和画图
        motion_energy = np.convolve(motion_energy, kernel, mode='same')
        
    return motion_energy

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_motion_energy_matplotlib(frame_indices, skeleton_motion_energy, radar_motion_energy, out_path, title):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ================= 1. 安全且兼容的无衬线字体后备名单 =================
    plt.rcParams.update({
        'font.family': 'sans-serif',
        # 我们把名字写全。在 Linux/服务器上，'Liberation Sans' 是官方的 Arial 无损平替
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',          # 强迫数学公式也使用干净的无衬线风格渲染
        'pdf.fonttype': 42                         # 必须保留！确保字体嵌入 PDF，排版绝不乱码
    })
    
    x = np.array(frame_indices)
    skeleton_color = ACADEMIC_COLORS["skeleton"]
    radar_color = ACADEMIC_COLORS["radar"]
    
    # ================= 2. 创建 1 行 2 列的并排大画布 =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    
    configs = [
        {
            "ax": ax1, 
            "data": np.array(skeleton_motion_energy), 
            "color": skeleton_color, 
            "label": "Skeleton motion energy"
        },
        {
            "ax": ax2, 
            "data": np.array(radar_motion_energy), 
            "color": radar_color, 
            "label": "Radar motion energy"
        }
    ]
    
    # ================= 3. 循环绘制并排子图 =================
    for cfg in configs:
        ax = cfg["ax"]
        
        # 绘制曲线
        ax.plot(x, cfg["data"], color=cfg["color"], linewidth=1.8, label=cfg["label"])
        
        # 自适应轴范围
        y_max = float(max(np.max(cfg["data"]), 1.0))
        ax.set_ylim(0, y_max * 1.05)
        ax.set_xlim(x[0], x[-1] if len(x) > 1 else x[0] + 1)
        
        # 标签和标题设置 (恢复普通的设置方式，不再依赖外部注册的对象)
        ax.set_title(cfg['label'], fontsize=12, pad=10, loc='left', color=(25/255, 25/255, 25/255))
        ax.set_xlabel("Frame", fontsize=10, color=(60/255, 60/255, 60/255))
        ax.set_ylabel("Motion Energy", fontsize=10, color=(60/255, 60/255, 60/255))
        
        # 样式和网格
        ax.grid(True, linestyle='-', linewidth=0.5, color=(224/255, 224/255, 224/255))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color((60/255, 60/255, 60/255))
        ax.spines['bottom'].set_color((60/255, 60/255, 60/255))
        
        # 图例
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    
    # ================= 4. 全局总标题及布局优化 =================
    fig.suptitle(title, fontsize=14, y=0.98, color=(25/255, 25/255, 25/255))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # ================= 5. 保存为一个并排的 PDF 矢量图 =================
    plt.savefig(out_path, format="pdf", bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot skeleton and radar motion energy for one aligned action segment."
    )
    parser.add_argument("--dataset-root", default="./Radar3DforRehab-master/dataset/dataset")
    parser.add_argument("--env", default="env1")
    parser.add_argument("--subject", default="subject26")
    parser.add_argument("--action", default="action12")
    parser.add_argument(
        "--out",
        default="motion_energy_env1_subject26_action12.pdf",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Use the same skeleton smoothing as ActionSegmenter before computing motion energy.",
    )
    args = parser.parse_args()

    action_dir = find_action_dir(args.dataset_root, args.env, args.subject, args.action)
    if not action_dir.exists():
        raise FileNotFoundError(f"Aligned action directory not found: {action_dir}")

    radar_frames, skeleton_frames = load_aligned_action(action_dir)
    if len(radar_frames) != len(skeleton_frames):
        raise ValueError(
            f"Frame count mismatch in {action_dir}: radar={len(radar_frames)}, skeleton={len(skeleton_frames)}"
        )

    skeleton_motion_energy = compute_skeleton_motion_energy(skeleton_frames, args.smooth_sigma)
    radar_motion_energy = compute_radar_motion_energy(radar_frames)
    frame_indices = np.arange(len(skeleton_motion_energy))

    title = f"{args.env} {args.subject} {args.action} aligned motion energy"
    plot_motion_energy_matplotlib(frame_indices, skeleton_motion_energy, radar_motion_energy, args.out, title)

    print(f"Saved motion energy plot to {Path(args.out).resolve()}")
    print(f"Frames: {len(frame_indices)}")


if __name__ == "__main__":
    main()

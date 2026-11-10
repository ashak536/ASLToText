"""
Generate synthetic ASL fingerspelling landmark data for all 26 static letters.

Each letter is defined by a canonical hand pose (21 MediaPipe landmarks × 3 coords),
then augmented with Gaussian noise to produce ~200 training samples per letter.
Output: training/data/asl_landmarks.csv
"""

import os
import csv
import random
import math

# ── MediaPipe landmark order ──────────────────────────────────────────
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

# Helper: build a 21-landmark pose from finger descriptions
# Each landmark is [x, y, z] relative to wrist (landmark 0 = [0,0,0])

def _extended_finger(mcp_x, mcp_y, length=0.35, spread=0.0, z=0.0):
    """Return 4 landmarks (MCP, PIP, DIP, TIP) for a finger pointing up."""
    seg = length / 3
    return [
        [mcp_x + spread * 0.0, mcp_y, z],
        [mcp_x + spread * 0.3, mcp_y - seg, z],
        [mcp_x + spread * 0.7, mcp_y - seg * 2, z],
        [mcp_x + spread * 1.0, mcp_y - seg * 3, z],
    ]

def _curled_finger(mcp_x, mcp_y, z=0.0):
    """Return 4 landmarks for a curled/fist finger."""
    return [
        [mcp_x, mcp_y, z],
        [mcp_x, mcp_y - 0.05, z - 0.03],
        [mcp_x - 0.01, mcp_y + 0.02, z - 0.05],
        [mcp_x - 0.02, mcp_y + 0.05, z - 0.04],
    ]

def _half_curled_finger(mcp_x, mcp_y, z=0.0):
    """Return 4 landmarks for a partially curled finger."""
    return [
        [mcp_x, mcp_y, z],
        [mcp_x, mcp_y - 0.08, z - 0.02],
        [mcp_x, mcp_y - 0.04, z - 0.06],
        [mcp_x + 0.01, mcp_y + 0.01, z - 0.05],
    ]

# Finger MCP base positions (x, y) — palm facing camera, right hand
THUMB_BASE = (-0.15, -0.05)
INDEX_BASE = (-0.08, -0.18)
MIDDLE_BASE = (0.0, -0.20)
RING_BASE = (0.08, -0.18)
PINKY_BASE = (0.15, -0.16)

def _thumb_across_curled():
    """Thumb tucked across curled fingers (like A, S, M, N, T)."""
    return [
        [-0.12, -0.04, 0.02],
        [-0.10, -0.08, 0.01],
        [-0.05, -0.10, 0.03],
        [0.0, -0.08, 0.04],
    ]

def _thumb_out_side():
    """Thumb sticking out to the side (like B, C, L)."""
    return [
        [-0.12, -0.04, 0.02],
        [-0.16, -0.08, 0.01],
        [-0.19, -0.10, 0.0],
        [-0.22, -0.08, 0.0],
    ]

def _thumb_up():
    """Thumb pointing up."""
    return [
        [-0.12, -0.04, 0.02],
        [-0.14, -0.10, 0.01],
        [-0.15, -0.18, 0.0],
        [-0.15, -0.26, 0.0],
    ]

def _thumb_index_touch():
    """Thumb tip touching index tip (like F, O, D-variant)."""
    return [
        [-0.12, -0.04, 0.02],
        [-0.10, -0.10, 0.01],
        [-0.08, -0.16, 0.02],
        [-0.06, -0.22, 0.02],  # close to index tip area
    ]

def _thumb_pinch():
    """Thumb touching index for pinch/circle."""
    return [
        [-0.12, -0.04, 0.02],
        [-0.10, -0.10, 0.02],
        [-0.06, -0.15, 0.04],
        [-0.04, -0.18, 0.04],
    ]


# ── Define canonical poses for each letter ────────────────────────────

def make_pose_A():
    """A: Fist with thumb alongside (thumb up beside index)."""
    wrist = [0, 0, 0]
    thumb = _thumb_up()
    index = _curled_finger(*INDEX_BASE)
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_B():
    """B: Flat hand, fingers together pointing up, thumb across palm."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _extended_finger(*INDEX_BASE, length=0.38, spread=0.0)
    middle = _extended_finger(*MIDDLE_BASE, length=0.40, spread=0.0)
    ring = _extended_finger(*RING_BASE, length=0.38, spread=0.0)
    pinky = _extended_finger(*PINKY_BASE, length=0.34, spread=0.0)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_C():
    """C: Curved hand like gripping a ball."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.16, -0.10, 0.03],
             [-0.18, -0.16, 0.04], [-0.18, -0.20, 0.04]]
    index = [[-0.08, -0.18, 0], [-0.10, -0.24, 0.04],
             [-0.08, -0.30, 0.06], [-0.05, -0.32, 0.06]]
    middle = [[0.0, -0.20, 0], [0.0, -0.26, 0.04],
              [0.02, -0.32, 0.06], [0.04, -0.33, 0.05]]
    ring = [[0.08, -0.18, 0], [0.08, -0.24, 0.04],
            [0.09, -0.30, 0.05], [0.10, -0.31, 0.04]]
    pinky = [[0.15, -0.16, 0], [0.14, -0.22, 0.03],
             [0.14, -0.27, 0.04], [0.14, -0.29, 0.03]]
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_D():
    """D: Index pointing up, other fingers curled, thumb touches middle."""
    wrist = [0, 0, 0]
    thumb = _thumb_pinch()
    index = _extended_finger(*INDEX_BASE, length=0.40, spread=0.0)
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_E():
    """E: All fingers curled, thumb across front."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _half_curled_finger(*INDEX_BASE)
    middle = _half_curled_finger(*MIDDLE_BASE)
    ring = _half_curled_finger(*RING_BASE)
    pinky = _half_curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_F():
    """F: Index+thumb circle, other three fingers extended/spread."""
    wrist = [0, 0, 0]
    thumb = _thumb_index_touch()
    index = _half_curled_finger(*INDEX_BASE)  # curled to touch thumb
    middle = _extended_finger(*MIDDLE_BASE, length=0.38, spread=0.02)
    ring = _extended_finger(*RING_BASE, length=0.36, spread=0.02)
    pinky = _extended_finger(*PINKY_BASE, length=0.32, spread=0.03)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_G():
    """G: Index pointing sideways, thumb parallel, others curled."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.18, -0.06, 0.01],
             [-0.24, -0.07, 0.0], [-0.30, -0.07, 0.0]]
    index = [[-0.08, -0.18, 0], [-0.16, -0.20, 0],
             [-0.24, -0.21, 0], [-0.32, -0.21, 0]]
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_H():
    """H: Index and middle pointing sideways, others curled."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = [[-0.08, -0.18, 0], [-0.16, -0.20, 0],
             [-0.24, -0.21, 0], [-0.32, -0.21, 0]]
    middle = [[0.0, -0.20, 0], [-0.08, -0.22, 0],
              [-0.16, -0.23, 0], [-0.24, -0.23, 0]]
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_I():
    """I: Pinky extended, others curled."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _curled_finger(*INDEX_BASE)
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _extended_finger(*PINKY_BASE, length=0.34, spread=0.02)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_J():
    """J: Like I but with motion (pinky extended). Static approximation."""
    # Same as I — motion letters get the static start pose
    return make_pose_I()

def make_pose_K():
    """K: Index up, middle angled out, thumb between them."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.10, -0.10, 0.02],
             [-0.06, -0.16, 0.04], [-0.02, -0.20, 0.04]]
    index = _extended_finger(*INDEX_BASE, length=0.38, spread=0.0)
    middle = [[-0.0, -0.20, 0], [0.04, -0.26, 0.02],
              [0.08, -0.30, 0.04], [0.12, -0.28, 0.04]]
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_L():
    """L: Index up, thumb out to side, forming L shape."""
    wrist = [0, 0, 0]
    thumb = _thumb_out_side()
    index = _extended_finger(*INDEX_BASE, length=0.40, spread=0.0)
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_M():
    """M: Three fingers (index, middle, ring) over thumb, curled down."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.08, -0.06, 0.04],
             [-0.04, -0.04, 0.06], [0.02, -0.02, 0.06]]
    index = _curled_finger(*INDEX_BASE, z=-0.02)
    middle = _curled_finger(*MIDDLE_BASE, z=-0.02)
    ring = _curled_finger(*RING_BASE, z=-0.02)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_N():
    """N: Two fingers (index, middle) over thumb, curled down."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.08, -0.06, 0.04],
             [-0.02, -0.04, 0.06], [0.06, -0.02, 0.06]]
    index = _curled_finger(*INDEX_BASE, z=-0.02)
    middle = _curled_finger(*MIDDLE_BASE, z=-0.02)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_O():
    """O: All fingertips touching thumb, forming O shape."""
    wrist = [0, 0, 0]
    thumb = _thumb_index_touch()
    index = [[-0.08, -0.18, 0], [-0.08, -0.22, 0.03],
             [-0.07, -0.24, 0.05], [-0.05, -0.22, 0.04]]
    middle = [[0.0, -0.20, 0], [0.0, -0.24, 0.03],
              [0.0, -0.24, 0.05], [-0.01, -0.22, 0.04]]
    ring = [[0.08, -0.18, 0], [0.06, -0.22, 0.03],
            [0.04, -0.24, 0.05], [0.02, -0.22, 0.04]]
    pinky = [[0.15, -0.16, 0], [0.12, -0.20, 0.02],
             [0.10, -0.22, 0.04], [0.08, -0.20, 0.03]]
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_P():
    """P: Like K but pointing down (wrist rotated)."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.14, -0.10, 0.02],
             [-0.12, -0.16, 0.04], [-0.08, -0.20, 0.04]]
    index = [[-0.08, -0.18, 0], [-0.12, -0.26, 0.02],
             [-0.14, -0.34, 0.02], [-0.14, -0.40, 0.01]]
    middle = [[0.0, -0.20, 0], [0.02, -0.28, 0.04],
              [0.06, -0.32, 0.06], [0.10, -0.30, 0.06]]
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_Q():
    """Q: Like G but pointing down — index and thumb down."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.14, 0.02, 0.01],
             [-0.16, 0.08, 0.0], [-0.18, 0.14, 0.0]]
    index = [[-0.08, -0.18, 0], [-0.10, -0.10, 0],
             [-0.12, -0.02, 0], [-0.14, 0.06, 0]]
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_R():
    """R: Index and middle crossed, others curled."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = [[-0.08, -0.18, 0], [-0.06, -0.26, 0.01],
             [-0.02, -0.34, 0.02], [0.02, -0.40, 0.02]]
    middle = [[0.0, -0.20, 0], [-0.02, -0.28, 0.01],
              [-0.06, -0.34, 0.02], [-0.08, -0.40, 0.02]]
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_S():
    """S: Fist with thumb across front of fingers."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _curled_finger(*INDEX_BASE, z=0.02)
    middle = _curled_finger(*MIDDLE_BASE, z=0.02)
    ring = _curled_finger(*RING_BASE, z=0.02)
    pinky = _curled_finger(*PINKY_BASE, z=0.02)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_T():
    """T: Fist with thumb between index and middle."""
    wrist = [0, 0, 0]
    thumb = [[-0.12, -0.04, 0.02], [-0.08, -0.08, 0.04],
             [-0.04, -0.10, 0.06], [-0.02, -0.08, 0.07]]
    index = _curled_finger(*INDEX_BASE, z=0.01)
    middle = _curled_finger(*MIDDLE_BASE, z=0.01)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_U():
    """U: Index and middle together pointing up, others curled."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _extended_finger(*INDEX_BASE, length=0.38, spread=0.0)
    middle = _extended_finger(*MIDDLE_BASE, length=0.40, spread=0.0)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_V():
    """V: Index and middle spread apart (peace sign)."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _extended_finger(*INDEX_BASE, length=0.38, spread=-0.06)
    middle = _extended_finger(*MIDDLE_BASE, length=0.40, spread=0.06)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_W():
    """W: Index, middle, ring spread apart pointing up."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = _extended_finger(*INDEX_BASE, length=0.38, spread=-0.05)
    middle = _extended_finger(*MIDDLE_BASE, length=0.40, spread=0.0)
    ring = _extended_finger(*RING_BASE, length=0.38, spread=0.05)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_X():
    """X: Index hooked (bent at DIP), others curled."""
    wrist = [0, 0, 0]
    thumb = _thumb_across_curled()
    index = [[-0.08, -0.18, 0], [-0.08, -0.26, 0],
             [-0.06, -0.30, 0.04], [-0.06, -0.26, 0.06]]
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _curled_finger(*PINKY_BASE)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_Y():
    """Y: Thumb and pinky extended, others curled."""
    wrist = [0, 0, 0]
    thumb = _thumb_out_side()
    index = _curled_finger(*INDEX_BASE)
    middle = _curled_finger(*MIDDLE_BASE)
    ring = _curled_finger(*RING_BASE)
    pinky = _extended_finger(*PINKY_BASE, length=0.34, spread=0.04)
    return [wrist] + thumb + index + middle + ring + pinky

def make_pose_Z():
    """Z: Index pointing up (static start — Z requires motion)."""
    # Same as D without thumb pinch; motion letter, static approximation
    return make_pose_D()


POSE_FUNCTIONS = {
    'A': make_pose_A, 'B': make_pose_B, 'C': make_pose_C, 'D': make_pose_D,
    'E': make_pose_E, 'F': make_pose_F, 'G': make_pose_G, 'H': make_pose_H,
    'I': make_pose_I, 'J': make_pose_J, 'K': make_pose_K, 'L': make_pose_L,
    'M': make_pose_M, 'N': make_pose_N, 'O': make_pose_O, 'P': make_pose_P,
    'Q': make_pose_Q, 'R': make_pose_R, 'S': make_pose_S, 'T': make_pose_T,
    'U': make_pose_U, 'V': make_pose_V, 'W': make_pose_W, 'X': make_pose_X,
    'Y': make_pose_Y, 'Z': make_pose_Z,
}


def normalize_pose(landmarks_21x3):
    """
    Normalize a 21-landmark pose to match the app.js toFeatures() pipeline:
    1. Subtract wrist position
    2. Scale by max distance from wrist
    3. Return flat 63-element list
    """
    wrist = landmarks_21x3[0]
    relative = []
    for pt in landmarks_21x3:
        relative.append([pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]])

    max_dist = max(math.hypot(*r) for r in relative) or 1.0
    flat = []
    for r in relative:
        flat.extend([round(r[0] / max_dist, 6),
                     round(r[1] / max_dist, 6),
                     round(r[2] / max_dist, 6)])
    return flat


def augment_pose(landmarks_21x3, noise_std=0.008):
    """Add small Gaussian noise to each coordinate."""
    noisy = []
    for pt in landmarks_21x3:
        noisy.append([
            pt[0] + random.gauss(0, noise_std),
            pt[1] + random.gauss(0, noise_std),
            pt[2] + random.gauss(0, noise_std),
        ])
    return noisy


def generate_dataset(samples_per_letter=200, noise_std=0.008):
    """Generate the full dataset and return as list of (label, features) tuples."""
    dataset = []
    for letter, pose_fn in sorted(POSE_FUNCTIONS.items()):
        canonical = pose_fn()
        assert len(canonical) == 21, f"Letter {letter}: expected 21 landmarks, got {len(canonical)}"
        for _ in range(samples_per_letter):
            augmented = augment_pose(canonical, noise_std)
            features = normalize_pose(augmented)
            assert len(features) == 63, f"Letter {letter}: expected 63 features, got {len(features)}"
            dataset.append((letter, features))
    return dataset


def save_csv(dataset, output_path):
    """Save dataset to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header = ['label'] + [f'f{i}' for i in range(63)]
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, features in dataset:
            writer.writerow([label] + features)
    print(f"Saved {len(dataset)} samples to {output_path}")


if __name__ == '__main__':
    random.seed(42)
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'asl_landmarks.csv')
    dataset = generate_dataset(samples_per_letter=200, noise_std=0.008)
    save_csv(dataset, output_path)
    print(f"\nGenerated data for {len(POSE_FUNCTIONS)} letters")
    print(f"Total samples: {len(dataset)}")
    # Summary
    from collections import Counter
    counts = Counter(label for label, _ in dataset)
    for letter in sorted(counts):
        print(f"  {letter}: {counts[letter]} samples")

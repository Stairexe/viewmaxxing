"""Fake Lyra 2.0: writes an 832x480 video of the anchor image, one frame per trajectory pose."""
import argparse, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
for f in ["--input_image_path", "--trajectory_path", "--experiment", "--checkpoint_dir", "--prompt", "--output_path"]:
    ap.add_argument(f)
ap.add_argument("--num_frames", type=int); ap.add_argument("--pose_scale", type=float); ap.add_argument("--num_samples", type=int)
ap.add_argument("--use_dmd", action="store_true")
a = ap.parse_args()
d = np.load(a.trajectory_path)
assert d["w2c"].shape[0] >= a.num_frames and np.allclose(d["w2c"][0], np.eye(4), atol=1e-5)
img = cv2.resize(cv2.imread(a.input_image_path), (832, 480))
stem = os.path.splitext(os.path.basename(a.input_image_path))[0]
vw = cv2.VideoWriter(os.path.join(a.output_path, stem + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (832, 480))
for i in range(a.num_frames):
    vw.write(np.roll(img, i * 3, axis=1))
vw.release()
print("fake Lyra:", a.num_frames, "frames, prompt:", a.prompt[:40], "dmd:", a.use_dmd)

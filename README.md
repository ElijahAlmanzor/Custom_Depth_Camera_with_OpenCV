# Stereo vision pipeline (7 steps)

This repo implements a simple stereo depth pipeline using two webcams and a printed ChArUco board. Each step has a script (`stepN_*.py`) you can run in order.

## Step 1: live preview (`step1_live_preview.py`)
Open both cameras and show them side-by-side to confirm camera indices, resolution, and frame rate are stable.

## Step 2: detect ChArUco (`step2_overlay_charuco.py`)
Detect ArUco markers + interpolate ChArUco corners on each live feed, and draw overlays to verify the board definition and print quality.

## Step 3: capture stereo pairs (`step3_save_images.py`)
Save synchronized left/right image pairs (and the detected corner data) when the board is visible well enough in both cameras.

## Step 4: calibrate intrinsics (`step4_get_intrinsics.py`)
Compute intrinsics + distortion for each camera separately from the saved ChArUco detections, and write `intrinsics_left.npz` / `intrinsics_right.npz`.

## Step 5: stereo calibrate + rectification (`step5_stereo_calibration.py`)
Using the two sets of intrinsics, estimate the transform between cameras and generate rectification outputs, saved to `stereo_calib.npz`.

## Step 6: rectified preview + epipolar lines (`step6_rectification_epipolar_lines.py`)
Apply undistort+rectify maps to live video and draw horizontal lines so you can visually check that corresponding points line up on the same rows.

## Step 7: disparity (`step7_disparity.py`)
Run a stereo matcher (e.g. SGBM) on the rectified images to produce a disparity map you can visualize.

## For slightly more detail, see LaTeX `stereo_vision_pipeline.tex`.

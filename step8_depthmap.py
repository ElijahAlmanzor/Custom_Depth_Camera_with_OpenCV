import cv2
import numpy as np


def main():
    camL_idx = 1
    camR_idx = 2

    # ---------- load calibration (steps 4-6 outputs) ----------
    intr_L = np.load("calib_results/intrinsics_left.npz")
    intr_R = np.load("calib_results/intrinsics_right.npz")
    stereo = np.load("calib_results/stereo_calib.npz")

    K_L = intr_L["K"]
    d_L = intr_L["dist"]
    K_R = intr_R["K"]
    d_R = intr_R["dist"]

    R1 = stereo["R1"]
    R2 = stereo["R2"]
    P1 = stereo["P1"]
    P2 = stereo["P2"]
    Q = stereo["Q"] if "Q" in stereo.files else None

    baseline = float(stereo["baseline"]) if "baseline" in stereo.files else None
    if baseline is None and "t" in stereo.files:
        baseline = float(np.linalg.norm(stereo["t"]))
    if baseline is None or baseline <= 0:
        raise RuntimeError("Stereo baseline is missing/invalid in calib_results/stereo_calib.npz")

    fx = float(P1[0, 0])
    if fx <= 0:
        raise RuntimeError("Invalid P1 matrix (fx <= 0) in calib_results/stereo_calib.npz")

    image_size = tuple(intr_L["image_size"])
    image_size_R = tuple(intr_R["image_size"])
    if image_size_R != image_size:
        raise RuntimeError(
            f"Left/right intrinsics image_size mismatch: left={image_size}, right={image_size_R}"
        )

    if "rms" in intr_L.files and "rms" in intr_R.files:
        print(
            f"Loaded intrinsics RMS: left={float(intr_L['rms']):.4f}, right={float(intr_R['rms']):.4f}"
        )
    if "stereo_rms" in stereo.files:
        print(f"--- TOTAL STEREO RMS ERROR: {float(stereo['stereo_rms']):.4f} ---")

    # ---------- camera setup ----------
    capL = cv2.VideoCapture(camL_idx, cv2.CAP_DSHOW)
    capR = cv2.VideoCapture(camR_idx, cv2.CAP_DSHOW)

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Could not open both cameras")

    width, height = image_size
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ---------- rectification maps ----------
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K_L, d_L, R1, P1, image_size, cv2.CV_16SC2
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K_R, d_R, R2, P2, image_size, cv2.CV_16SC2
    )

    # ---------- StereoSGBM ----------
    min_depth_m = 0.5
    max_depth_m = 6.0

    # Choose a disparity range that can represent at least down to min_depth_m.
    # d_max ~= fx * B / Z_min.
    disp_max = (fx * baseline) / min_depth_m
    num_disparities = int(16 * np.ceil(disp_max / 16.0))  # must be multiple of 16
    num_disparities = max(num_disparities, 16 * 4)
    num_disparities = min(num_disparities, 16 * 30)  # keep compute reasonable
    block_size = 5               # odd, 3–11 typical

    stereo_sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print("Depth running (press q to quit)")
    print(f"Depth display range: {min_depth_m:.2f}m .. {max_depth_m:.2f}m")
    print(f"Using baseline={baseline:.4f} m, fx={fx:.2f} px, num_disparities={num_disparities}")
    display_scale = 0.5

    print("Waiting for first frames...")
    frameL = None
    frameR = None
    for _ in range(60):
        okL, fL = capL.read()
        okR, fR = capR.read()
        if okL and okR:
            frameL, frameR = fL, fR
            break
        cv2.waitKey(10)

    if frameL is None or frameR is None:
        raise RuntimeError("Failed to read initial frames from both cameras")

    live_size = (frameL.shape[1], frameL.shape[0])
    if live_size != image_size:
        raise RuntimeError(f"Live resolution {live_size} does not match calibration {image_size}")

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

        disparity = stereo_sgbm.compute(grayL, grayR).astype(np.float32) / 16.0

        valid = disparity > 0.0

        if Q is not None:
            xyz = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
            depth = xyz[:, :, 2].astype(np.float32)
        else:
            # Depth from disparity (Z = f * B / d). Units: metres if B is metres.
            depth = np.full_like(disparity, np.nan, dtype=np.float32)
            depth[valid] = (fx * baseline) / disparity[valid]

        # visualisation: near = hot, far = cold
        depth_clip = np.clip(depth, min_depth_m, max_depth_m)
        depth_vis = (max_depth_m - depth_clip) / (max_depth_m - min_depth_m)
        depth_vis = (depth_vis * 255.0).astype(np.uint8)
        depth_vis[~valid] = 0
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        left_vis = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([left_vis, depth_color])

        cv2.putText(
            combined,
            "Left (rectified)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined,
            "Depth (m)",
            (left_vis.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined_disp = cv2.resize(
            combined,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("Left | Depth map", combined_disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

## PLOT THE RECTIFIED IMAGES - VRERIFY THAT THEY'RE ACTUALLY CORRECT!


if __name__ == "__main__":
    main()









# -------------------------------------------------------------------------
# NOTE: Using object detection to reduce stereo matching search space
#
# The classical stereo pipeline computes dense disparity for every pixel
# in the rectified left image by matching sliding windows along epipolar
# lines in the right image using StereoSGBM.
#
# This is computationally expensive because:
#   • Every pixel is evaluated
#   • For each pixel, numDisparities candidate matches are tested
#   • Each match compares blockSize × blockSize patches
#
# Object detection (e.g. YOLO) can be used to reduce this cost by focusing
# stereo matching only on regions of interest (ROIs), such as detected
# strawberries. This does NOT remove the correspondence problem, it only
# reduces the search space.
#
# Two valid design options exist:
#
# A. ROI constrained dense stereo (StereoSGBM retained)
#
#   • Run YOLO on left and right images to obtain bounding boxes or masks
#   • Define ROIs in the rectified images where depth is required
#   • Run StereoSGBM only inside these ROIs
#   • Optionally mask or ignore disparity outside ROIs
#
#   Properties:
#     • Same geometry and depth equation (Z = fB / d)
#     • Same sliding window matching and SGBM optimisation
#     • Dense depth inside ROIs only
#     • Reduced computation proportional to ROI area
#
#   This is the simplest drop in optimisation and preserves the classical
#   stereo pipeline.
#
# B. ROI guided sparse stereo (StereoSGBM replaced)
#
#   • Run YOLO to identify objects of interest
#   • Detect and describe keypoints only inside ROIs
#     (e.g. ORB, SIFT, SuperPoint)
#   • Match descriptors between left and right images
#   • Triangulate matched keypoints using known stereo geometry
#
#   Properties:
#     • Sparse depth at feature locations only
#     • No dense disparity image
#     • Much lower computation
#     • Requires robust feature matching
#
#   In this case, StereoSGBM is no longer used. Depth is obtained via
#   triangulation rather than dense disparity.
#
# Choice between A and B depends on whether dense depth within the object
# region is required, or whether sparse depth points are sufficient.
# -------------------------------------------------------------------------

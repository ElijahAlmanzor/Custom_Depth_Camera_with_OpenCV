import cv2
import numpy as np
from pathlib import Path


# -------------------------
# ChArUco board definition
# -------------------------

squares_x = 5
squares_y = 7
square_length = 0.04   # metres
marker_length = 0.03   # metres

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)


def main():
    base = Path("stereo_data")
    charuco_dir = base / "charuco"

    # ---------- load intrinsics ----------
    intr_L = np.load("calib_results/intrinsics_left.npz")
    intr_R = np.load("calib_results/intrinsics_right.npz")

    K_L = intr_L["K"]
    d_L = intr_L["dist"]
    K_R = intr_R["K"]
    d_R = intr_R["dist"]

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

    # If step4 saved metadata, sanity-check that step5's board definition matches.
    if "squares_x" in intr_L.files:
        meta = (
            int(intr_L["squares_x"]),
            int(intr_L["squares_y"]),
            float(intr_L["square_length"]),
            float(intr_L["marker_length"]),
        )
        here = (squares_x, squares_y, square_length, marker_length)
        if meta != here:
            raise RuntimeError(
                "Board definition mismatch vs saved intrinsics: "
                f"step4={meta}, step5={here}"
            )

    # ---------- load stereo correspondences ----------
    image_paths = sorted((base / "left").glob("img_*.png"))
    if not image_paths:
        raise RuntimeError(f"No images found under {base / 'left'}")

    sample = cv2.imread(str(image_paths[0]))
    if sample is None:
        raise RuntimeError(f"Failed to read sample image: {image_paths[0]}")
    sample_size = (sample.shape[1], sample.shape[0])
    if sample_size != image_size:
        raise RuntimeError(
            f"Saved images size {sample_size} does not match intrinsics image_size {image_size}"
        )

    obj_points = []
    img_points_L = []
    img_points_R = []

    used = 0

    chessboard_corners = board.getChessboardCorners()  # (N, 3)

    for img_path in image_paths:
        idx = img_path.stem.split("_")[1]
        meta_path = charuco_dir / f"data_{idx}.npz"

        if not meta_path.exists():
            continue

        data = np.load(meta_path)

        cornersL = data["charucoCornersL"]
        idsL = data["charucoIdsL"]
        cornersR = data["charucoCornersR"]
        idsR = data["charucoIdsR"]

        if cornersL is None or idsL is None or cornersR is None or idsR is None:
            continue

        idsL_flat = idsL.flatten()
        idsR_flat = idsR.flatten()
        common_ids = np.intersect1d(idsL_flat, idsR_flat)

        if len(common_ids) < 10:
            continue

        objp = []
        imgpL = []
        imgpR = []

        for cid in common_ids:
            iL = np.where(idsL_flat == cid)[0][0]
            iR = np.where(idsR_flat == cid)[0][0]

            objp.append(chessboard_corners[cid])
            imgpL.append(cornersL[iL][0])
            imgpR.append(cornersR[iR][0])

        obj_points.append(np.asarray(objp, dtype=np.float32))
        img_points_L.append(np.asarray(imgpL, dtype=np.float32))
        img_points_R.append(np.asarray(imgpR, dtype=np.float32))

        used += 1

    if used < 10:
        raise RuntimeError(f"Not enough valid stereo frames ({used})")

    print(f"Using {used} stereo frames")

    # ---------- stereo calibration ----------
    # Match the distortion model used in step4 (intrinsics). If you calibrated with
    # CALIB_RATIONAL_MODEL, you must include it here too, otherwise OpenCV may ignore
    # the higher-order coefficients and stereo calibration can blow up.
    flags = cv2.CALIB_FIX_INTRINSIC

    saved_flags_L = int(intr_L["calib_flags"]) if "calib_flags" in intr_L.files else None
    saved_flags_R = int(intr_R["calib_flags"]) if "calib_flags" in intr_R.files else None
    if saved_flags_L is not None and saved_flags_R is not None and saved_flags_L != saved_flags_R:
        raise RuntimeError(
            f"Left/right intrinsics flags mismatch: left={saved_flags_L}, right={saved_flags_R}"
        )

    if saved_flags_L is not None:
        flags |= (saved_flags_L & cv2.CALIB_RATIONAL_MODEL)
    elif d_L.size > 5 or d_R.size > 5:
        flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6
    )

    rms, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_L,
        img_points_R,
        K_L, d_L,
        K_R, d_R,
        image_size,
        criteria=criteria,
        flags=flags
    )

    print(f"--- TOTAL STEREO RMS ERROR: {rms:.4f} ---")
    print("Rotation R:\n", R)
    print("Translation t (m):\n", t.T)

    baseline = np.linalg.norm(t)
    print(f"Baseline (m): {baseline:.4f}")

    # ---------- rectification ----------
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_L, d_L,
        K_R, d_R,
        image_size,
        R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # ---------- save everything ----------
    out = Path("calib_results")
    out.mkdir(exist_ok=True)

    np.savez(
        out / "stereo_calib.npz",
        stereo_rms=rms,
        R=R,
        t=t,
        baseline=baseline,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=roi1,
        roi2=roi2
    )

    print("Saved stereo calibration to calib_results/stereo_calib.npz")


if __name__ == "__main__":
    main()

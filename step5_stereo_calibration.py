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

    # ---------- load stereo correspondences ----------
    image_paths = sorted((base / "left").glob("img_*.png"))

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

        if cornersL is None or cornersR is None:
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
    flags = cv2.CALIB_FIX_INTRINSIC

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

    print(f"Stereo RMS error: {rms:.4f}")
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

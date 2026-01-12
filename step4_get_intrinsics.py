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


def calibrate_camera(camera_name, image_dir, charuco_dir, out_dir):
    print(f"\n=== Calibrating {camera_name} camera ===")

    image_paths = sorted(image_dir.glob("img_*.png"))

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    used = 0

    for img_path in image_paths:
        idx = img_path.stem.split("_")[1]
        meta_path = charuco_dir / f"data_{idx}.npz"

        if not meta_path.exists():
            continue

        data = np.load(meta_path)

        if camera_name == "left":
            corners = data["charucoCornersL"]
            ids = data["charucoIdsL"]
        else:
            corners = data["charucoCornersR"]
            ids = data["charucoIdsR"]

        if corners is None or ids is None:
            continue

        if len(ids) < 10:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if image_size is None:
            image_size = img.shape[1], img.shape[0]

        all_charuco_corners.append(corners)
        all_charuco_ids.append(ids)
        used += 1

    if used < 10:
        raise RuntimeError(
            f"Not enough valid frames for {camera_name} intrinsics ({used})"
        )

    print(f"Using {used} frames")

    flags = cv2.CALIB_RATIONAL_MODEL

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags
    )

    print(f"RMS reprojection error: {rms:.4f}")
    print("Camera matrix K:\n", K)
    print("Distortion coeffs:\n", dist.ravel())

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"intrinsics_{camera_name}.npz"

    np.savez(
        out_path,
        K=K,
        dist=dist,
        image_size=image_size,
        rms=rms
    )

    print(f"Saved intrinsics to {out_path}")


def main():
    base = Path("stereo_data")
    charuco_dir = base / "charuco"

    calibrate_camera(
        "left",
        base / "left",
        charuco_dir,
        Path("calib_results")
    )

    calibrate_camera(
        "right",
        base / "right",
        charuco_dir,
        Path("calib_results")
    )


if __name__ == "__main__":
    main()

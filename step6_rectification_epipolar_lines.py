import cv2
import numpy as np
import pickle
from pathlib import Path


def draw_epipolar_lines(img, step=40, colour=(0, 255, 0)):
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), colour, 1)
    return img


def main():
    camL_idx = 1
    camR_idx = 2

    # ---------- load calibration ----------
    pkl_path = Path("stereo_calibration.pkl")
    use_prebuilt_maps = False

    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            calib = pickle.load(f)
        mapL1 = calib["map1_l"]
        mapL2 = calib["map2_l"]
        mapR1 = calib["map1_r"]
        mapR2 = calib["map2_r"]
        h, w = mapL1.shape[:2]
        calib_size = (w, h)
        use_prebuilt_maps = True
        stereo_rms = None
        print("Loaded rectification maps from stereo_calibration.pkl")
    else:
        try:
            intr_L = np.load("calib_results/intrinsics_left.npz")
            intr_R = np.load("calib_results/intrinsics_right.npz")
            stereo = np.load("calib_results/stereo_calib.npz")
        except FileNotFoundError as e:
            raise RuntimeError(
                "Missing calibration file(s). Expected either "
                "`stereo_calibration.pkl` or the trio "
                "`calib_results/intrinsics_left.npz`, `calib_results/intrinsics_right.npz`, "
                "and `calib_results/stereo_calib.npz`."
            ) from e

        K_L = intr_L["K"]
        d_L = intr_L["dist"]
        K_R = intr_R["K"]
        d_R = intr_R["dist"]

        R1 = stereo["R1"]
        R2 = stereo["R2"]
        P1 = stereo["P1"]
        P2 = stereo["P2"]

        calib_size = tuple(intr_L["image_size"])
        stereo_rms = float(stereo["stereo_rms"]) if "stereo_rms" in stereo else None
        print("Loaded calibration from calib_results/*.npz")

    capL = cv2.VideoCapture(camL_idx, cv2.CAP_DSHOW)
    capR = cv2.VideoCapture(camR_idx, cv2.CAP_DSHOW)

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Could not open both cameras")

    width, height = calib_size
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

    hL, wL = frameL.shape[:2]
    hR, wR = frameR.shape[:2]

    print("Live left size :", wL, hL)
    print("Live right size:", wR, hR)
    print("Calibration size:", calib_size)
    if stereo_rms is not None:
        print(f"--- TOTAL STEREO RMS ERROR: {stereo_rms:.4f} ---")

    if (wL, hL) != calib_size:
        raise RuntimeError("Live resolution does not match calibration")

    if not use_prebuilt_maps:
        # build maps AFTER size confirmation
        mapL1, mapL2 = cv2.initUndistortRectifyMap(
            K_L, d_L, R1, P1, (wL, hL), cv2.CV_16SC2
        )

        mapR1, mapR2 = cv2.initUndistortRectifyMap(
            K_R, d_R, R2, P2, (wR, hR), cv2.CV_16SC2
        )

        print("Rectification maps built correctly")
    print("Press q to quit, s to swap cameras")

    display_scale = 0.6
    line_step = 40

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            cv2.waitKey(10)
            continue

        rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

        combined = np.hstack([rectL, rectR])
        draw_epipolar_lines(combined, step=line_step)

        cv2.putText(
            combined,
            "Check: do objects sit on the SAME green line? (q=quit, s=swap)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        combined_disp = cv2.resize(combined, None, fx=display_scale, fy=display_scale)
        cv2.imshow("Rectified stereo diagnostic", combined_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            capL, capR = capR, capL
            print("Cameras swapped.")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np


def main():
    camL_idx = 1
    camR_idx = 2

    capL = cv2.VideoCapture(camL_idx, cv2.CAP_DSHOW)
    capR = cv2.VideoCapture(camR_idx, cv2.CAP_DSHOW)

    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Could not open both cameras")

    width, height = 1280, 720
    for cap in (capL, capR):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # ---------- load calibration ----------
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

    image_size = tuple(intr_L["image_size"])

    # ---------- rectification maps ----------
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K_L, d_L, R1, P1, image_size, cv2.CV_16SC2
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K_R, d_R, R2, P2, image_size, cv2.CV_16SC2
    )

    # ---------- StereoSGBM ----------
    num_disparities = 16 * 8     # must be multiple of 16
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

    print("Disparity running (press q to quit)")

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

        # visualisation
        disp_vis = cv2.normalize(
            disparity, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX
        )
        disp_vis = disp_vis.astype(np.uint8)

        combined = np.hstack([
            cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
        ])

        cv2.imshow("Left (rectified) | Disparity", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

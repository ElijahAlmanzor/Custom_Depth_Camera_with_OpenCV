import cv2
import numpy as np


def draw_epipolar_lines(img, step=40, colour=(0, 255, 0)):
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), colour, 1)
    return img


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

    print("Waiting for first frames...")

    okL, frameL = capL.read()
    okR, frameR = capR.read()

    if not okL or not okR:
        raise RuntimeError("Failed to read initial frames")

    hL, wL = frameL.shape[:2]
    hR, wR = frameR.shape[:2]

    print("Live left size :", wL, hL)
    print("Live right size:", wR, hR)
    print("Calibration size:", tuple(intr_L["image_size"]))

    if (wL, hL) != tuple(intr_L["image_size"]):
        raise RuntimeError("Live resolution does not match calibration")

    # build maps AFTER size confirmation
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K_L, d_L, R1, P1, (wL, hL), cv2.CV_16SC2
    )

    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K_R, d_R, R2, P2, (wR, hR), cv2.CV_16SC2
    )

    print("Rectification maps built correctly")
    print("Press q to quit")

    display_scale = 0.6

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

        rectL = draw_epipolar_lines(rectL)
        rectR = draw_epipolar_lines(rectR)

        stereo = np.hstack([rectL, rectR])
        stereo_disp = cv2.resize(
            stereo, None, fx=display_scale, fy=display_scale
        )

        cv2.imshow("Rectified stereo", stereo_disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

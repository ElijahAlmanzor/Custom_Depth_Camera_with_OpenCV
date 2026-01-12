import cv2
import numpy as np
import os
from pathlib import Path


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

    # -------- ChArUco board --------
    squaresX, squaresY = 5, 7
    squareLength = 0.04
    markerLength = 0.03

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength,
        markerLength,
        aruco_dict
    )

    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # -------- Output folders --------
    base_dir = Path("stereo_data")
    left_dir = base_dir / "left"
    right_dir = base_dir / "right"
    charuco_dir = base_dir / "charuco"

    for d in (left_dir, right_dir, charuco_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_idx = len(list(left_dir.glob("*.png")))

    print("Stereo ChArUco data capture")
    print("Press SPACE to save a stereo pair")
    print("Press q to quit")

    MIN_CHARUCO_CORNERS = 15
    display_scale = 0.6   # <<< adjust if needed

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        validL = False
        validR = False

        charucoCornersL = charucoIdsL = None
        charucoCornersR = charucoIdsR = None

        # -------- LEFT --------
        cornersL, idsL, _ = aruco_detector.detectMarkers(grayL)
        if idsL is not None:
            cv2.aruco.drawDetectedMarkers(frameL, cornersL, idsL)
            retL, charucoCornersL, charucoIdsL = cv2.aruco.interpolateCornersCharuco(
                cornersL, idsL, grayL, board
            )
            if retL > MIN_CHARUCO_CORNERS:
                validL = True
                cv2.aruco.drawDetectedCornersCharuco(
                    frameL, charucoCornersL, charucoIdsL
                )

        # -------- RIGHT --------
        cornersR, idsR, _ = aruco_detector.detectMarkers(grayR)
        if idsR is not None:
            cv2.aruco.drawDetectedMarkers(frameR, cornersR, idsR)
            retR, charucoCornersR, charucoIdsR = cv2.aruco.interpolateCornersCharuco(
                cornersR, idsR, grayR, board
            )
            if retR > MIN_CHARUCO_CORNERS:
                validR = True
                cv2.aruco.drawDetectedCornersCharuco(
                    frameR, charucoCornersR, charucoIdsR
                )

        status = "VALID" if (validL and validR) else "INVALID"
        colour = (0, 255, 0) if status == "VALID" else (0, 0, 255)

        cv2.putText(frameL, f"L {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        cv2.putText(frameR, f"R {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

        # -------- DISPLAY (RESIZED COPY ONLY) --------
        stereo_full = np.hstack([frameL, frameR])
        stereo_disp = cv2.resize(
            stereo_full,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("Stereo Capture", stereo_disp)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if validL and validR:
                left_path = left_dir / f"img_{img_idx:03d}.png"
                right_path = right_dir / f"img_{img_idx:03d}.png"
                meta_path = charuco_dir / f"data_{img_idx:03d}.npz"

                cv2.imwrite(str(left_path), frameL)
                cv2.imwrite(str(right_path), frameR)

                np.savez(
                    meta_path,
                    charucoCornersL=charucoCornersL,
                    charucoIdsL=charucoIdsL,
                    charucoCornersR=charucoCornersR,
                    charucoIdsR=charucoIdsR
                )

                print(f"Saved stereo pair {img_idx:03d}")
                img_idx += 1
            else:
                print("Rejected frame, ChArUco not valid in both cameras")

        elif key == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

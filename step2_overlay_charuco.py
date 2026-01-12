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

    print("ChArUco overlay running (stable API). Press q to quit.")

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        # -------- left --------
        cornersL, idsL, _ = aruco_detector.detectMarkers(grayL)
        if idsL is not None:
            cv2.aruco.drawDetectedMarkers(frameL, cornersL, idsL)

            retL, charucoCornersL, charucoIdsL = \
                cv2.aruco.interpolateCornersCharuco(
                    cornersL, idsL, grayL, board
                )

            if retL > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    frameL, charucoCornersL, charucoIdsL
                )

        # -------- right --------
        cornersR, idsR, _ = aruco_detector.detectMarkers(grayR)
        if idsR is not None:
            cv2.aruco.drawDetectedMarkers(frameR, cornersR, idsR)

            retR, charucoCornersR, charucoIdsR = \
                cv2.aruco.interpolateCornersCharuco(
                    cornersR, idsR, grayR, board
                )

            if retR > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    frameR, charucoCornersR, charucoIdsR
                )

        # -------- display --------
        h = 720
        frameL = cv2.resize(frameL, (int(frameL.shape[1]*h/frameL.shape[0]), h))
        frameR = cv2.resize(frameR, (int(frameR.shape[1]*h/frameR.shape[0]), h))
        cv2.imshow("Stereo ChArUco Overlay", np.hstack([frameL, frameR]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

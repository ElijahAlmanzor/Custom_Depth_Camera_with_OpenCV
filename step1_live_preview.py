import cv2
import numpy as np
import time


def try_open(index, width=1280, height=720, fps=30):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DirectShow tends to behave better on Windows
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # grab a frame to confirm it actually streams
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None

    return cap


def main():
    # probe a few likely indices
    candidates = list(range(0, 6))
    caps = {}
    for idx in candidates:
        cap = try_open(idx)
        if cap is not None:
            caps[idx] = cap
            print(f"Opened camera index {idx}")

    if len(caps) < 2:
        print("Could not open at least two cameras.")
        print("Try different USB ports, unplug one then the other, or close apps like Teams, OBS, Camera app.")
        for cap in caps.values():
            cap.release()
        return

    # pick first two indices found
    indices = sorted(caps.keys())[:2]
    capL = caps[indices[0]]
    capR = caps[indices[1]]
    print(f"Using indices: left={indices[0]}, right={indices[1]}")

    # release any extra cameras we opened
    for idx in list(caps.keys()):
        if idx not in indices:
            caps[idx].release()

    last_t = time.time()

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()

        if not okL or frameL is None:
            frameL = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frameL, "Left camera read failed", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if not okR or frameR is None:
            frameR = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frameR, "Right camera read failed", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # resize to same height for stacking
        h = 720
        frameL = cv2.resize(frameL, (int(frameL.shape[1] * (h / frameL.shape[0])), h))
        frameR = cv2.resize(frameR, (int(frameR.shape[1] * (h / frameR.shape[0])), h))

        # overlay FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - last_t))
        last_t = now

        cv2.putText(frameL, f"L idx {indices[0]}  FPS {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frameR, f"R idx {indices[1]}  FPS {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        stacked = np.hstack([frameL, frameR])
        cv2.imshow("Stereo preview (press q to quit)", stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

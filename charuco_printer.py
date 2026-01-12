import cv2
import cv2.aruco as aruco
import numpy as np


def main():
    # ---------- paper setup ----------
    dpi = 300
    paper_width_m = 0.210   # A4 width
    paper_height_m = 0.297  # A4 height

    px_per_m = dpi / 0.0254
    paper_w_px = int(paper_width_m * px_per_m)
    paper_h_px = int(paper_height_m * px_per_m)

    # white A4 canvas
    canvas = np.ones((paper_h_px, paper_w_px), dtype=np.uint8) * 255

    # ---------- charuco board ----------
    squaresX = 5
    squaresY = 7
    squareLength = 0.04    # metres
    markerLength = 0.03    # metres

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength,
        markerLength,
        aruco_dict
    )

    # board size in pixels (exact physical scaling)
    board_w_m = squaresX * squareLength
    board_h_m = squaresY * squareLength

    board_w_px = int(board_w_m * px_per_m)
    board_h_px = int(board_h_m * px_per_m)

    board_img = board.generateImage(
        (board_w_px, board_h_px),
        marginSize=0,
        borderBits=1
    )

    # ---------- center board ----------
    y0 = (paper_h_px - board_h_px) // 2
    x0 = (paper_w_px - board_w_px) // 2

    canvas[y0:y0 + board_h_px, x0:x0 + board_w_px] = board_img

    # ---------- save ----------
    filename = "charuco_A4_centered_300dpi.png"
    cv2.imwrite(filename, canvas)

    print(f"Saved {filename}")
    print("Print instructions:")
    print("- Paper size: A4")
    print("- Scale: 100% (do NOT fit to page)")
    print("- Measure a square: it must be exactly 40 mm")

    # preview
    preview = cv2.resize(canvas, (paper_w_px // 4, paper_h_px // 4))
    cv2.imshow("A4 ChArUco (preview)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

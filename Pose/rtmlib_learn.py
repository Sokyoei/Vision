"""
rtmlib: https://github.com/Tau-J/rtmlib
"""

import cv2
from rtmlib import Wholebody, draw_skeleton


def main():
    device = 'cpu'  # cpu, cuda, mps
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    img = cv2.imread(r'D:\Download\7Q18l-dtisZ21T3cS1kw-16o.jpg')

    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    wholebody = Wholebody(
        to_openpose=openpose_skeleton,
        mode='balanced',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
        backend=backend,
        device=device,
    )

    keypoints, scores = wholebody(img)

    # visualize
    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)
    img_show = draw_skeleton(img, keypoints, scores, kpt_thr=0.5)

    cv2.namedWindow("rtmlib", cv2.WINDOW_FREERATIO)
    cv2.imshow('rtmlib', img_show)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

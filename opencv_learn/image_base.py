from pathlib import Path

import cv2

ROOT = Path(".").absolute().parent
img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"

window = cv2.namedWindow("Popstar Ahri", cv2.WINDOW_FREERATIO)
src = cv2.imread(rf"{img_path}")
cv2.imshow("Popstar Ahri", src)
cv2.waitKey(0)
cv2.destroyAllWindows()

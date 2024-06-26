import cv2

USE_SIFT = 1
USE_ORB = 0

if cv2.__version__ >= "4.5.5":
    img = cv2.imread(r"../data/Ahri/Popstar Ahri.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if USE_SIFT:
        model = cv2.SIFT_create()
    elif USE_ORB:
        model = cv2.ORB_create(20000)
    else:
        raise NotImplementedError("please use SIFT or ORB")

    keypoints = model.detect(gray, None)
    cv2.drawKeypoints(img, keypoints, img, (0, 255, 0))

    cv2.namedWindow("keypoints", cv2.WINDOW_FREERATIO)
    cv2.imshow("keypoints", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

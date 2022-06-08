# database set

camera_capture = cv2.VideoCapture(0)

file_num = 1

while (camera_capture.isOpened()):
    ret_flag, image = camera_capture.read()
    cv2.imshow('Captured Image', image)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('s'):
        cv2.imwrite(str(file_num) + ".name" + ".jpg", image)
        print("Successfully saved " + str(file_num) + ".jpg !")
        print("=" * 60)
        file_num += 1
    elif k == ord(' '):
        break

camera_capture.release()
cv2.destroyAllWindows()
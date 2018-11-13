import cv2
import numpy as np

global k
global xy
xy = np.zeros((4, 2))
k = 0


def openPKimage(path):
    img = cv2.imread(path)

    return img


def onmouse(event, x, y, flags, param):
    set_label = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        global k
        global xy
        global f, f1, f2
        xy[k] = x, y
        print(xy)
        k = k + 1
        print(k)
        if k == 4:
            k = 0
            # if cv2.waitKey() == ord('1'):
            #     label = '1'
            #
            # if cv2.waitKey() == ord('2'):
            #     label = '0'

            while not set_label:
                if cv2.waitKey() == ord('1'):
                    label = '1'
                    set_label = 1
                    print('label is ', label)
                elif cv2.waitKey() == ord('2'):
                    label = '0'
                    set_label = 1
                    print('label is ', label)
                else:
                    print('set a label please(1 = 1, 2 = 0)')
            for i in xy:
                # f.write(str(i) + '\n' + "lable: " + label + '\n\n')
                f.write(str(i)[1:len(str(i))-1] + ' ')
                f1.write(str(i)[1:len(str(i))-1] + '\n')
            f.write(label + '\n')
            f2.write(label+'\n')
            xy = np.zeros((4, 2))


def saveloctofile(img):
    """
    Save position information to files.

    :param img: load image which need to be marked position.
    :return: 3 files which contain some coordinates information.
    """
    global f, f1, f2
    f = open('./location_information/7jpg.txt', 'a')
    f1 = open('./location_information/7jpgcoordinate.txt', 'a')
    f2 = open('./location_information/7jpglabel.txt', 'a')
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", onmouse)
    cv2.imshow("img", img)

    while True:
        cv2.imshow("img", img)
        if cv2.waitKey() == ord('q'):
            break  # press q to end
    cv2.destroyAllWindows()
    f.close()
    f1.close()
    f2.close()


def main():
    img = openPKimage('C:/Users/Haoyu/Desktop/haoyu/6.jpg')

    saveloctofile(img)


if __name__ == '__main__':
    main()

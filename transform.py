from Librarys import *


def order_points(pts):
    """
    Initialize a list of coordinates that wil be ordered.

    top-left
    top-right
    bottom-right
    bottom-left

    :param pts any four points
    :return ordered four points like above order.

    """
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points, the top-right
    # point will have the smallest difference, whereas the
    # bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # print(rect)
    return rect


def four_point_transform(image, pts):
    """
    :param image: image we want to transform
    :param pts: four points
    :return:
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    widthB = np.sqrt(((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    # print(widthA,widthB,maxWidth)

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))
    # print(heightA,heightB,maxHeight)
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view".
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32"
    )

    # print('start to transform')
    # compute the perspective transform matrix and the apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # print('transform end')
    return warped


def transform_from_file(coordinate_path, lable_path, image_path, save_image_path):

    allpts = np.loadtxt(coordinate_path)
    labels = np.loadtxt(lable_path)
    print(allpts.shape)
    print(allpts.shape[0])
    num = int(allpts.shape[0]/4)
    pts = np.reshape(allpts, (num, 4, 2))
    print(pts.shape)
    print(pts[0])

    img = cv2.imread(image_path)

    for step, pts4 in enumerate(pts):
        warped = four_point_transform(img, pts4)
        if labels[step] == 0:
            cv2.imwrite(save_image_path + 'Empty/' + str(step) + '.jpg', warped)
        else:
            cv2.imwrite(save_image_path + 'Occupied/' + str(step)+'.jpg', warped)


if __name__ == '__main__':
    # pts = np.zeros((4,2))
    # pts = np.loadtxt('coordinate.txt')
    # img = cv2.imread('111.jpg')
    # warped = four_point_transform(img, pts)
    #
    # cv2.imshow('original',iQmg)
    # cv2.imshow('warped', warped)
    # cv2.waitKey(0)
    path = './locationinformation/testcoordinate.txt'
    image_path = './pklotImage/images.jpeg'
    save_image_path = './pklotImageSeg/'
    lable_path = '/home/haoyu/pklot/pklotsss/locationinformation/testlabel.txt'
    transform_from_file(path, lable_path, image_path, save_image_path)
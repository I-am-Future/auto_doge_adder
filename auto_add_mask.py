import dlib
import cv2
import numpy as np

def auto_scale(img):
    h, w, d = img.shape
    if w < 1920:
        return 1
    else:
        return 1920 / w

# kernelSize = 1
# kernel = np.ones((kernelSize, kernelSize))

# preprocess the foreground picture
IMG_FG = cv2.imread('doge.jpg')
IMG_FG_ORIGINAL = IMG_FG.copy()
IMG_FG_GRAY = cv2.cvtColor(IMG_FG, cv2.COLOR_BGR2GRAY)
IMG_FG_THRESHED = cv2.adaptiveThreshold(IMG_FG_GRAY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
# IMG_FG_THRESHED = cv2.morphologyEx(IMG_FG_THRESHED, cv2.MORPH_OPEN, kernel)
# IMG_FG_THRESHED = cv2.morphologyEx(IMG_FG_THRESHED, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(IMG_FG_THRESHED, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_areas = []
for contour in contours:
    contour_areas.append((contour, cv2.contourArea(contour)))
contour_areas.sort(key=lambda x: x[1], reverse=True)
wanted_contour = contour_areas[0][0]

IMG_FG_SHAPE = IMG_FG_GRAY.shape
FG_MASK = np.zeros(IMG_FG_SHAPE)
FG_MASK = cv2.fillPoly(FG_MASK, [wanted_contour], 255)
# preprocess complete. now we have the mask of the fore image

# preprocess the background picture
img_bg = cv2.imread('source.JPG')
img_bg_shape = img_bg.shape
final = img_bg.copy()

detector = dlib.get_frontal_face_detector()

back_img_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
faces = detector(back_img_gray)
for index, face in enumerate(faces):
    # initialization for each face process
    fg_init = np.zeros(img_bg_shape, dtype=np.uint8)
    temp_bg_mask = np.zeros(img_bg_shape[:2], dtype=np.uint8)
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    width = x2 - x1
    height = y2 - y1
    mask_width = int(1.55 * width)
    if mask_width % 2 != 0:
        mask_width -= 1
    mask_height = int(1.55 * height)
    if mask_height % 2 != 0:
        mask_height -= 1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    # print(mask_width, mask_height)
    fg_left = 0 if centerX - mask_width // 2 >= 0 else (mask_width // 2 - centerX)
    fg_top = 0 if centerY - mask_height // 2 >= 0 else mask_height // 2 - centerY
    fg_right = mask_width if img_bg_shape[1] - (centerX + mask_width // 2) >= 0 else mask_width - (centerX + mask_width // 2 - img_bg_shape[1])
    fg_bottom = mask_height if img_bg_shape[0] - (centerY + mask_height // 2) >= 0 else mask_height - (centerY + mask_height // 2 - img_bg_shape[0])
    scaled_fg = cv2.resize(IMG_FG_ORIGINAL, (mask_width, mask_height))
    scaled_fg_mask = cv2.resize(FG_MASK, (mask_width, mask_height))
    # print(fg_left, fg_right, fg_top, fg_bottom)
    roi_left = max(0, centerX - mask_width // 2)
    roi_right = min(img_bg_shape[1], centerX + mask_width // 2)
    roi_top = max(0, centerY - mask_height // 2)
    roi_bottom = min(img_bg_shape[0], centerY + mask_height // 2)
    # print(roi_left, roi_right, roi_top, roi_bottom)

    temp_bg_mask[roi_top:roi_bottom, roi_left:roi_right] = scaled_fg_mask[fg_top:fg_bottom, fg_left:fg_right].copy()
    fg_init[roi_top:roi_bottom, roi_left:roi_right] = scaled_fg[fg_top:fg_bottom, fg_left:fg_right].copy()

    temp_bg_mask = cv2.bitwise_not(temp_bg_mask)
    temp_fg_mask = cv2.bitwise_not(temp_bg_mask)
    final = cv2.bitwise_or(final, final, mask=temp_bg_mask)
    fg_init = cv2.bitwise_or(fg_init, fg_init, mask=temp_fg_mask)
    final = cv2.add(final, fg_init)
    print(f'{index+1}/{len(faces)} faces complete.')

scale = auto_scale(final)
display_image = cv2.resize(final, None, None, scale, scale)

cv2.imshow('preview', display_image)
cv2.waitKey(0)
cv2.imwrite('output.jpg', final)

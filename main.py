# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np




cam = cv2.VideoCapture('Lane Detection Test Video-01.mp4')


def main():
    first_loop = 1

    last_left_top_y = 0
    last_left_top_x = 0

    last_left_bottom_y = 0
    last_left_bottom_x = 0

    last_right_top_y = 0
    last_right_top_x = 0

    last_right_bottom_y = 0
    last_right_bottom_x = 0

    while True:
        first_loop = 0

        ret, frame = cam.read()

        if ret is False:
            break

        frame = cv2.resize(frame, (344, 240))
        frame_copy = frame.copy()

        # 3
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 4
        trapezoid = np.zeros(frame.shape, dtype=np.uint8)

        w = frame.shape[1]
        h = frame.shape[0]

        upper_right = (w * 0.56, h * 0.74)
        upper_left = (w * 0.43, h * 0.74)
        lower_left = (w * 0.14, h)
        lower_right = (w * 0.85, h)

        trarr = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
        cv2.fillConvexPoly(trapezoid, trarr, 1)

        trapezoid = trapezoid * frame
        cv2.imshow('Trapezoid', trapezoid)

        framearr = np.array([(w, 0), (0, 0), (0, h), (w, h)])

        trarr = np.float32(trarr)
        framearr = np.float32(framearr)

        perspective = cv2.getPerspectiveTransform(trarr, framearr)

        stretched = cv2.warpPerspective(trapezoid, perspective, (w, h))
        #cv2.imshow('Streched', stretched)

        #6
        blurred = cv2.blur(stretched, ksize=(7, 7))
        cv2.imshow('Blurred', blurred)

        #7
        div = 3
        sobel_vertical = np.float32([-1/div, -2/div, -1/div, 0, 0, 0, 1/div, 2/div, 1/div])
        sobel_horizontal = np.transpose(sobel_vertical)

        vertical = cv2.filter2D(np.float32(blurred), -1, sobel_vertical)
        vertical_frame = cv2.convertScaleAbs(vertical)
        horizontal = cv2.filter2D(np.float32(blurred), -1, sobel_horizontal)
        horizontal_frame = cv2.convertScaleAbs(horizontal)

        cv2.imshow('Vertical', vertical_frame)
        cv2.imshow('Horizontal', horizontal_frame)

        sobel = np.sqrt(vertical ** 2 + horizontal ** 2)
        sobel = cv2.convertScaleAbs(sobel)

        cv2.imshow('Sobel Filter', sobel)

        #8
        ret, binary = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Image', binary)

        #9
        frame2 = binary.copy()
        wsmall = np.int32(w / 30)
        hsmall = np.int32(h / 2.5)
        frame2[0:h-hsmall, 0:wsmall] = 0
        frame2[0:np.int32(h/2), 0:wsmall*2] = 0
        frame2[0:h, w-wsmall*6:w] = 0

        #cv2.imshow('Frame2', frame2)
        left = np.argwhere(frame2[0:h, 0:np.int32(w/2)]>1)
        right = np.argwhere(frame2[0:h, np.int32(w/2):w]>1)

        left_xs = left[:, 1]
        left_ys = left[:, 0]
        right_xs = right[:, 1]
        right_ys = right[:, 0]

        # 10
        if left_xs.size > 0 and left_ys.size > 0:
            left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)

        if right_xs.size > 0 and right_ys.size > 0:
            right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        b_l = left_line[0]
        a_l = left_line[1]

        b_r = right_line[0]
        a_r = right_line[1]

        left_top_y = 0
        left_bottom_y = h
        right_top_y = 0
        right_bottom_y = h

        left_top_x = -b_l / a_l
        left_bottom_x = (h - b_l) / a_l
        right_top_x = -b_r / a_r
        right_bottom_x = (h - b_r) / a_r

        if first_loop == 0:

            if -(10 ** 8) < left_top_x < 10**8:
                last_left_top_x = -b_l / a_l

            if -(10 ** 8) < left_bottom_x < 10**8:
                last_left_bottom_x = left_bottom_x

            if -(10 ** 8) < right_top_x < 10**8:
                last_right_top_x = right_top_x

            if -(10 ** 8) < right_bottom_x < 10**8:
                last_right_bottom_x = right_bottom_x

            last_left_top_y = left_top_y
            last_left_bottom_y = left_bottom_y
            last_right_top_y = right_top_y
            last_right_bottom_y = right_bottom_y

        cv2.line(binary, [np.int32(last_left_top_x), np.int32(last_left_top_y)], [np.int32(last_left_bottom_x), np.int32(last_left_bottom_y)], (200, 0, 0), 5)
        cv2.line(binary, [np.int32(last_right_top_x+w/2), np.int32(last_right_top_y)], [np.int32(last_right_bottom_x+w/2), np.int32(last_right_bottom_y)], (100, 0, 0), 5)

        cv2.imshow('Linii!', binary)

        # 11
        final_l = np.zeros(frame.shape, dtype=np.uint8)
        cv2.line(final_l, [np.int32(last_left_top_x), np.int32(last_left_top_y)],
                 [np.int32(last_left_bottom_x), np.int32(last_left_bottom_y)], (255, 0, 0), 3)

        perspective2 = cv2.getPerspectiveTransform(framearr, trarr)
        final_l = cv2.warpPerspective(final_l, perspective2, (w, h))
        cv2.imshow('final l', final_l)

        left_points = np.argwhere(final_l > 0)

        final_r = np.zeros(frame.shape, dtype=np.uint8)
        cv2.line(final_r, [np.int32(last_right_top_x + w / 2), np.int32(last_right_top_y)],
                 [np.int32(last_right_bottom_x + w / 2), np.int32(last_right_bottom_y)], (50, 250, 50), 5)

        perspective2 = cv2.getPerspectiveTransform(framearr, trarr)
        final_r = cv2.warpPerspective(final_r, perspective2, (w, h))
        cv2.imshow('final r', final_r)

        right_points = np.argwhere(final_r)

        final = frame_copy.copy()

        for coord in left_points:
            x, y = coord
            final[x, y] = (255, 0, 0)

        for coord in right_points:
            x, y = coord
            final[x, y] = (50, 250, 50)

        cv2.imshow('Lane Detector!', final)

    cam.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

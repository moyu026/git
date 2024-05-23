import cv2 as cv
import os
import numpy as np

img_point_folder = 'img_2/pts'
img_point_list = os.listdir(img_point_folder)

img_model_folder = 'img_2/obj'
img_model_list = os.listdir(img_model_folder)

img_result_folder = 'img_result'

def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    return cv_img


indx = -1
count = 0
with open('img_result/side_part.txt', 'w') as f:
    for img_points in img_point_list:

        img_point_path = os.path.join(img_point_folder, img_points)

        img_point = cv_imread(img_point_path)
        img_point = img_point[250:760, 520:940]
        # gray_point = cv.cvtColor(img_point, cv.COLOR_RGB2GRAY)
        theight, twidth =img_point.shape[:2]

        img_name = []
        mini = []
        result = {}

        for img_models in img_model_list:
            img_model_path = os.path.join(img_model_folder, img_models)

            img_model = cv_imread(img_model_path)
            img_model = img_model[85:910, 320:1760]
            # gray_model = cv.cvtColor(img_model, cv.COLOR_RGB2GRAY)

            result = cv.matchTemplate(img_model, img_point, cv.TM_SQDIFF_NORMED)

            # cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)

            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

            strmin_val = str(min_val)

            cv.rectangle(img_model, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)

            # cv.imshow("MatchResult----MatchingValue=" + strmin_val, img_model)
            # cv.waitKey(0)

            mini.append(min_val)
            img_name.append(img_models)


        result = dict(zip(img_name, mini))
        best_result = min(result.items(), key=lambda x: x[1])

        indx = indx + 1
        name = list(enumerate(result))


        if name[indx][1] == best_result[0]:
            f.write(str(result) + '\n')
            f.write('indx_name' + str(name[indx][1]) + ',' + str(result[name[indx][1]]) + '\n')
            f.write(str(best_result) + '\n')
            f.write('\n')

            count = count + 1
        elif name[indx][1] != best_result[0]:
            print('==================================')
            f.write('==================================' + '\n')
            f.write(str(result) + '\n')
            f.write('indx_name' + str(name[indx][1]) + ',' + str(result[name[indx][1]]) + '\n')
            f.write(str(best_result) + '\n')
            f.write('\n')

        # f.write(str(name[indx][1]) + '\n')
        # f.write(str(result) + '\n')
        # f.write('indx_name' + str(name[indx][1]) + ',' + str(result[name[indx][1]]) + '\n')
        # f.write(str(best_result) + '\n')

        print(result)
        print(best_result)
        print('indx_name:', name[indx][1],result[name[indx][1]])
        print('best_name:', best_result[0])


print(count)

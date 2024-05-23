import cv2 as cv
import os

img_point_path = 'img/BLQ_5687_A_PTS2.png'

img_point = cv.imread(img_point_path)
img_point = img_point[250:760, 520:940]
# gray_point = cv.cvtColor(img_point, cv.COLOR_RGB2GRAY)
theight, twidth =img_point.shape[:2]


img_model_folder = 'img/obj_threshold'

img_model_list = os.listdir(img_model_folder)
mini = []
img_name = []

result = {}

for img_models in img_model_list:
    img_model_path = os.path.join(img_model_folder, img_models)

    img_model = cv.imread(img_model_path)
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

print(result)
print(best_result)




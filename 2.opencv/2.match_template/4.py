import os
import cv2 as cv




def point_model_template(img_point_top_folder, img_model_top_folder, side=False):
    if side:
        img_point_top_list = os.listdir(img_point_top_folder)
        img_model_top_list = os.listdir(img_model_top_folder)
    else:
        img_point_top_list = os.listdir(img_point_top_folder)
        img_model_top_list = os.listdir(img_model_top_folder)

    indx = -1
    count = 0
    point_name = []
    true_result = []
    template_result = []

    # 读取point_top图片
    for img_point_tops in img_point_top_list:
        img_point_tops_path = os.path.join(img_point_top_folder, img_point_tops)
        img_point_top = cv.imread(img_point_tops_path)

        # 选取point_top图片模型区域
        img_point_top = img_point_top[250:760, 520:940]
        theight, twidth = img_point_top.shape[:2]

        img_model_top_name = []
        template_result_top = []
        result_top = {}



        # 读取model_top图片
        for img_model_tops in img_model_top_list:
            img_model_tops_path = os.path.join(img_model_top_folder, img_model_tops)
            img_model_top = cv.imread(img_model_tops_path)
            # 选取model_top图片模型区域
            img_model_top = img_model_top[85:910, 320:1760]

            # 用cv.matchTemplate匹配point_top图片区域与model_top图片区域
            result_tops = cv.matchTemplate(img_model_top, img_point_top, cv.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result_tops)  # 获取结果中的最小值，最大值，以及最值的位置

            #在img_model图片上画出匹配区域
            strmin_val = str(min_val)
            cv.rectangle(img_model_top, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)

            # cv.imshow("MatchResult----MatchingValue=" + strmin_val, img_model_top)
            # cv.waitKey(0)

            template_result_top.append(min_val)  # img_point和每一张img_model图片匹配的值
            img_model_top_name.append(img_model_tops)

        result_top = dict(zip(img_model_top_name, template_result_top))
        best_result = min(result_top.items(), key=lambda x: x[1])

        indx = indx + 1
        name = list(enumerate(result_top))

        # if name[indx][1] == best_result[0]:
        #     count = count + 1
        # elif name[indx][1] != best_result[0]:
        #     print('==================================')



        # print(result_top)

        # print('点云：', img_point_tops)
        # print('真实结果：', name[indx][1], result_top[name[indx][1]])
        # print('匹配结果：', best_result)



        point_name.append(name[indx][1])
        true_result.append(result_top[name[indx][1]])
        template_result.append(best_result)


    # print(point_name)

        # if result_top[name[indx][1]] == best_result[1]:
        #     print('点云：', img_point_tops)
        #     print('真实结果：', name[indx][1], result_top[name[indx][1]])
        #     print('匹配结果：', best_result)
        #
        # elif result_top[name[indx][1]] - best_result[1] < 0.2:
        #     print('不匹配')



    return point_name, true_result, template_result








        # print(result_top[name[indx][1]]) # 点云图片与真实模型的结果
        # print(best_result)   # 点云模型与匹配模型的结果
        #
        # print('indx_name', name[indx][1])
        # print('best_name', best_result[0])



    # print(count)



if __name__ == '__main__':

    img_point_top_folder = 'img/pts_test/top'
    img_point_side_folder = 'img/pts_test/side'

    img_model_top_folder = 'img/obj_test/top'
    img_model_side_folder = 'img/obj_test/side'


    # point_model_template(img_point_top_folder, img_model_top_folder)
    # point_model_template(img_point_side_folder, img_model_side_folder)




    point_name_top, true_result_top, template_result_top = point_model_template(img_point_top_folder, img_model_top_folder)
    for i in range(len(point_name_top)):
        if true_result_top[i] == template_result_top[i][1]:
                print('点云-top：', point_name_top[i])
                print('真实结果：', point_name_top[i], true_result_top[i])
                print('匹配结果：', template_result_top[i])
        else:
            print('不匹配', point_name_top[i])


            point_name_side, true_result_side, template_result_side = point_model_template(img_point_side_folder, img_model_side_folder)
            for j in range(len(point_name_side)):
                if true_result_side[j] == template_result_side[j][1]:
                    print('点云-side：', point_name_side[j])
                    print('真实结果：', point_name_side[j], true_result_side[j])
                    print('匹配结果：', template_result_side[j])






# true_result, template_result = point_model_template(img_point_side_folder, img_model_side_folder)

# if true_result =





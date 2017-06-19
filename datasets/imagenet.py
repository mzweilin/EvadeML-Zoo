import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import os
from multiprocessing import Pool
from keras.preprocessing import image

from models.keras_models import keras_resnet50_imagenet_model
from models.keras_models import keras_vgg19_imagenet_model
from models.keras_models import keras_inceptionv3_imagenet_model

pool = Pool()

corrected_idx = [4, 6, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 34, 35, 37, 41, 42, 44, 46, 51, 52, 53, 54, 55, 57, 59, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 88, 91, 92, 94, 95, 97, 98, 100, 101, 103, 104, 106, 107, 108, 110, 112, 115, 117, 119, 121, 122, 124, 125, 126, 128, 129, 130, 132, 134, 137, 138, 140, 141, 142, 143, 144, 148, 151, 152, 153, 154, 156, 157, 159, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 181, 182, 185, 186, 187, 188, 191, 192, 193, 195, 196, 197, 198, 199, 200, 201, 205, 207, 208, 210, 211, 212, 214, 215, 216, 217, 218, 220, 224, 226, 227, 229, 232, 233, 234, 236, 237, 242, 244, 245, 246, 247, 248, 249, 250, 252, 253, 255, 257, 258, 263, 265, 267, 270, 271, 275, 277, 279, 280, 281, 282, 283, 286, 287, 288, 293, 297, 298, 299, 303, 304, 306, 307, 308, 310, 313, 315, 317, 318, 319, 320, 321, 322, 323, 325, 327, 328, 329, 331, 332, 334, 335, 337, 339, 340, 341, 342, 345, 346, 347, 348, 351, 352, 353, 354, 355, 356, 358, 360, 361, 362, 363, 365, 366, 367, 368, 369, 370, 372, 373, 374, 376, 380, 381, 382, 383, 385, 387, 389, 390, 391, 392, 393, 394, 395, 396, 398, 400, 401, 402, 404, 405, 406, 408, 410, 411, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 427, 428, 429, 430, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446, 447, 448, 450, 451, 452, 453, 454, 457, 459, 460, 461, 462, 463, 465, 466, 469, 470, 471, 473, 474, 475, 477, 478, 479, 481, 482, 483, 484, 485, 486, 488, 489, 490, 492, 493, 494, 496, 497, 498, 499, 500, 501, 503, 504, 505, 506, 510, 511, 512, 513, 514, 519, 521, 523, 524, 525, 526, 527, 528, 530, 532, 534, 535, 536, 537, 539, 540, 541, 543, 544, 545, 546, 547, 549, 550, 551, 553, 555, 558, 559, 562, 566, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 581, 582, 584, 585, 586, 587, 589, 590, 591, 594, 596, 598, 599, 601, 603, 604, 606, 608, 609, 612, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 629, 630, 632, 633, 635, 637, 638, 639, 640, 644, 645, 647, 648, 649, 650, 652, 654, 656, 659, 661, 662, 664, 667, 669, 671, 672, 673, 674, 675, 678, 680, 681, 684, 685, 688, 689, 690, 691, 692, 693, 696, 699, 701, 702, 704, 705, 708, 709, 710, 712, 713, 715, 717, 718, 719, 721, 723, 724, 725, 727, 728, 729, 730, 731, 732, 733, 734, 735, 737, 738, 739, 740, 741, 743, 745, 746, 748, 751, 752, 754, 755, 756, 760, 761, 762, 764, 765, 766, 768, 769, 770, 772, 773, 774, 777, 778, 780, 783, 785, 786, 787, 789, 791, 792, 793, 795, 796, 797, 805, 806, 810, 811, 813, 814, 818, 819, 820, 825, 828, 829, 830, 831, 832, 833, 835, 836, 838, 839, 841, 843, 845, 846, 848, 849, 851, 852, 853, 854, 856, 857, 858, 859, 860, 863, 864, 865, 866, 867, 870, 871, 872, 873, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 887, 889, 890, 891, 892, 894, 895, 897, 899, 902, 903, 904, 905, 906, 907, 908, 909, 910, 912, 914, 915, 918, 919, 920, 921, 922, 923, 924, 926, 927, 928, 929, 930, 931, 933, 934, 935, 938, 940, 942, 943, 944, 946, 949, 950, 951, 955, 956, 959, 960, 961, 962, 963, 964, 967, 968, 969, 970, 971, 973, 974, 975, 978, 980, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 999, 1000, 1001, 1002, 1003, 1004, 1006, 1007, 1008, 1009, 1011, 1013, 1015, 1017, 1019, 1020, 1021, 1022, 1025, 1027, 1028, 1029, 1030, 1032, 1033, 1035, 1036, 1038, 1039, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1057, 1058, 1059, 1060, 1062, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1072, 1073, 1074, 1076, 1077, 1080, 1081, 1082, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1093, 1096, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1106, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1126, 1128, 1129, 1130, 1131, 1133, 1134, 1135, 1137, 1138, 1140, 1142, 1143, 1144, 1145, 1147, 1148, 1151, 1152, 1156, 1157, 1158, 1160, 1162, 1163, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1179, 1180, 1181, 1183, 1184, 1186, 1187, 1188, 1189, 1192, 1193, 1194, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1208, 1209, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1219, 1220, 1222, 1224, 1225, 1226, 1227, 1229, 1230, 1231, 1232, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1247, 1249, 1250, 1251, 1254, 1255, 1256, 1258, 1260, 1261, 1263, 1269, 1270, 1271, 1273, 1274, 1275, 1276, 1279, 1281, 1282, 1283, 1284, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1295, 1297, 1299, 1300, 1301, 1302, 1304, 1305, 1306, 1307, 1308, 1312, 1315, 1316, 1321, 1322, 1324, 1325, 1327, 1328, 1329, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1341, 1342, 1343, 1344, 1346, 1348, 1349, 1350, 1351, 1352, 1356, 1358, 1359, 1360, 1361, 1363, 1364, 1365, 1367, 1369, 1372, 1373, 1375, 1376, 1378, 1379, 1381, 1382, 1383, 1384, 1386, 1388, 1389, 1390, 1393, 1394, 1395, 1397, 1398, 1400, 1403, 1404, 1405, 1406, 1409, 1412, 1415, 1417, 1422, 1424, 1425, 1426, 1427, 1429, 1430, 1431, 1432, 1433, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1446, 1448, 1449, 1450, 1452, 1453, 1455, 1457, 1458, 1459, 1460, 1461, 1462, 1464, 1465, 1467, 1468, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1482, 1484, 1485, 1486, 1488, 1489, 1491, 1492, 1494, 1496, 1497, 1498, 1499]


def load_single_image(img_path, img_size=224):
    size = (img_size,img_size)
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Embeded preprocessing in the model.
    # x = preprocess_input(x)
    return x

def _load_single_image(args):
    img_path, img_size = args
    return load_single_image(img_path, img_size)

def data_imagenet(img_folder, img_size, label_style = 'caffe', label_size = 1000, selected_idx = None):
    fnames = os.listdir(img_folder)
    fnames = sorted(fnames, key = lambda x: int(x.split('.')[1]))
    
    if isinstance(selected_idx, list):
        selected_fnames = [fnames[i] for i in selected_idx]
    elif isinstance(selected_idx, int):
        selected_fnames = fnames[:selected_idx]
    else:
        selected_fnames = fnames

    labels = map(lambda x: int(x.split('.')[0]), selected_fnames)
    img_path_list = map(lambda x: [os.path.join(img_folder, x), img_size], selected_fnames)
    X = map(_load_single_image, img_path_list)
    X = np.concatenate(X, axis=0)
    Y = np.eye(1000)[labels]
    return X, Y


class ImageNetDataset:
    def __init__(self):
        self.dataset_name = "ImageNet"
        # self.image_size = 224
        self.num_channels = 3
        self.num_classes = 1000
        self.img_folder = "/tmp/ILSVRC2012_img_val_labeled_caffe"

    def get_test_dataset(self, img_size=224, num_images=100):
        self.image_size = img_size
        X, Y = data_imagenet(self.img_folder, self.image_size, selected_idx=num_images)
        X /= 255
        return X, Y

    def get_test_data(self, img_size, idx_begin, idx_end):
        # Return part of the dataset.
        self.image_size = img_size
        X, Y = data_imagenet(self.img_folder, self.image_size, selected_idx=range(idx_begin, idx_end))
        X /= 255
        return X, Y

    def load_model_by_name(self, model_name, logits=False, input_range_type=1, input_tensor = None):
        """
        :params logits: no softmax layer if True.
        :params scaling: expect [-0.5,0.5] input range if True, otherwise [0, 1]
        """
        if model_name == 'resnet50':
            model = keras_resnet50_imagenet_model(logits=logits, input_range_type=input_range_type)
        elif model_name == 'vgg19':
            model = keras_vgg19_imagenet_model(logits=logits, input_range_type=input_range_type)
        elif model_name == 'inceptionv3':
            model = keras_inceptionv3_imagenet_model(logits=logits, input_range_type=input_range_type)

        return model

if __name__ == '__main__':
    # label_style = 'caffe'
    # # img_folder = "/mnt/nfs/taichi/imagenet_data/data_val_labeled_%s" % label_style
    # img_folder = "/tmp/ILSVRC2012_img_val_labeled_caffe"
    # X, Y = data_imagenet(img_folder, selected_idx=10)
    # print (X.shape)
    # print (np.argmax(Y, axis=1))

    dataset = ImageNetDataset()

    X, Y = dataset.get_test_dataset()
    model = dataset.load_model_by_name('ResNet50')


    


# @Time    : 2021/07/15 22:41
# @Author  : SY.M
# @FileName: path_config.py

project_result_root = 'E:/PyCharmProjects/transformer on time series/Gated_Transfomer_for_TSC/result/'


class MTS:
    # dataset archive root
    root = 'E:/PyCharmWorkSpace/dataset/MTS_dataset'

    # choose file
    # file_name = 'ArabicDigits'
    file_name = 'ECG'

    # path to save result(Default, depend on up two parameters)
    result_root = project_result_root + f'for_MTS/MTS_archive/{file_name}'


class UEA:
    # dataset archive root
    root = 'E:/PyCharmWorkSpace/dataset/UEA/raw'

    # choose file
    file_name = 'ArticularyWordRecognition'
    file_name = 'BasicMotions'

    # path to save result(Default, depend on up two parameters)
    result_root = project_result_root + f'result/for_MTS/UEA_archive/{file_name}'


class UCR:
    # dataset archive root
    root = 'E:/PyCharmWorkSpace/dataset/UCRArchive_2018'

    # choose file
    # file_name = 'Coffee'
    file_name = 'Beef'
    # file_name = 'MedicalImages'

    # path to save result(Default, depend on up two parameters)
    result_root = project_result_root + f'for_TS/UCR_archive/{file_name}'

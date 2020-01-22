import math


def filter(thresh, out_df_all, out_df_behavior, out_df_per, out_cate):
    start = 0
    end = 0
    count_mal = 0
    count_ben = 0
    with open('features_raw.txt', 'r') as file:
        row = file.readline()
        spl = row.split(';')
        for i in range(len(spl)):
            if 'NOT_USED_NATIVE_LIB' == spl[i]:
                start = i
                print(i)
            if 'NOT_INSTALL_ANOTHER_PACKAGE_SERVICE' == spl[i]:
                end = i
                print(i)

    with open('features_raw.txt', 'r') as file:
        dict = {}
        dict2 = {}
        dict_per = {}
        dict2_per = {}
        with open(out_df_all, 'w') as dff:
            for row in file:
                # print(row)
                spl = row.replace('\n', '').split(';')
                if spl[1] == '0':
                    count_mal += 1
                else:
                    count_ben += 1
                for i in range(start, end + 1):
                    # print(spl[i])
                    if not spl[i].startswith('NOT'):
                        if spl[i] + '_MAL' not in dict and spl[i] + '_BEN' not in dict:
                            dict[spl[i] + '_MAL'] = 0
                            dict[spl[i] + '_BEN'] = 0
                            dict2[spl[i]] = 0
                        if spl[1] == '0':
                            dict[spl[i] + '_MAL'] += 1
                        else:
                            dict[spl[i] + '_BEN'] += 1

                for i in range(end + 1, len(spl)):
                    if spl[i] + '_MAL' not in dict_per:
                        dict_per[spl[i] + '_MAL'] = 0
                        dict_per[spl[i] + '_BEN'] = 0
                        dict2_per[spl[i]] = 0
                    if spl[1] == '0':
                        dict_per[spl[i] + '_MAL'] += 1
                    else:
                        dict_per[spl[i] + '_BEN'] += 1

            with open(out_df_behavior, 'w') as f:
                for key in dict2.keys():
                    if math.fabs(float(dict[key + '_MAL']) / float(count_mal) - float(dict[key + '_BEN']) / float(count_ben)) > thresh \
                        and (float(dict[key + '_MAL']) / float(count_mal) + float(dict[key + '_BEN']) / float(count_ben)) > 0.1:
                        f.write(key + ';' + str(float(dict[key + '_MAL']) / float(count_mal)) + ';' + str(float(dict[key + '_BEN']) / float(count_ben)) + ';' +
                                str(float(dict[key + '_MAL']) / float(count_mal) - float(dict[key + '_BEN']) / float(count_ben)) + '\n')
                        with open(out_cate, 'a') as f123:
                            f123.write(key + '\n')

            with open(out_df_per, 'w') as f:
                for key in dict2_per.keys():
                    if math.fabs(float(dict_per[key + '_MAL']) / float(count_mal) - float(dict_per[key + '_BEN']) / float(count_ben)) > thresh \
                        and (float(dict_per[key + '_MAL']) / float(count_mal) + float(dict_per[key + '_BEN']) / float(count_ben)) > 0.1:
                        f.write(key + ';' + str(float(dict_per[key + '_MAL']) / float(count_mal)) + ';' + str(float(dict_per[key + '_BEN']) / float(count_ben)) + ';' +
                                str(float(dict_per[key + '_MAL']) / float(count_mal) - float(dict_per[key + '_BEN']) / float(count_ben)) + '\n')
                        with open(out_cate, 'a') as f123:
                            f123.write(key + '\n')

            for key in dict.keys():
                if key.endswith('_MAL'):
                    dff.write(key + ': ' + str(dict[key]) + '/' + str(count_mal) + '>>>>>>>>>>>>' + str(float(dict[key]) / float(count_mal)) + '\n')
                else:
                    dff.write(key + ': ' + str(dict[key]) + '/' + str(count_ben) + '>>>>>>>>>>>>' + str(float(dict[key]) / float(count_ben)) + '\n')

            for key in dict_per.keys():
                if key.endswith('_MAL'):
                    dff.write(key + ': ' + str(dict_per[key]) + '/' + str(count_mal) + '>>>>>>>>>>>>' + str(float(dict_per[key]) / float(count_mal)) + '\n')
                else:
                    dff.write(key + ': ' + str(dict_per[key]) + '/' + str(count_ben) + '>>>>>>>>>>>>' + str(float(dict_per[key]) / float(count_ben)) + '\n')

    print(count_ben)
    print(count_mal)


def features(out_feature, out_cate):
    with open(out_feature, 'w') as f:
        l_fea = []
        f.write('Package;Class;SIZE;NUI;NCLASS')
        with open(out_cate, 'r') as f_cate:
            for fea in f_cate:
                if fea != 'SERVICE_INSIDE':
                    f.write(';' + fea.replace('\n', ''))
                    l_fea.append(fea.replace('\n', ''))
        f.write('\n')

        with open('features_raw.txt', 'r') as f_raw:
            for row in f_raw:
                spl = row.replace('\n', '').split(';')
                f.write(spl[0] + ';' + spl[1] + ';' + str(spl[2]) + ';' + str(spl[3]) + ';' + str(spl[4]))

                for i in range(0, 25):
                    if 'NOT_' + l_fea[i] not in spl:
                        f.write(';1')
                    else:
                        f.write(';0')
                for i in range(25, len(l_fea)):
                    if l_fea[i] in spl:
                        f.write(';1')
                    else:
                        f.write(';0')
                f.write('\n')



if __name__ == '__main__':
    filter(0.1, 'data/out_df_all.txt', 'data/out_df_behavior.txt', 'data/out_df_per.txt', 'data/out_cate.txt')
    features('data/out_features.txt', 'data/out_cate.txt')

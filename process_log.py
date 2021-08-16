import os
import ast

list1=[0, 2, 3]
list2=[0, 1, 2, 5]

def read_file(filename):
    info = filename[11:]
    remark=''
    if info[0] == '0':
        remark += '全连接层 '
    elif info[0] == '2':
        remark += 'Leaky Relu '
    elif info[0] == '3':
        remark += 'PRelu '

    if info[3] == '0':
        remark += '不删边 '
    else:
        if info[1] == '1':
            remark += 'curvature从低到高mask '
        if info[1] == '2':
            remark += 'curvature从高到低mask '
        if info[3] == '1':
            remark += 'rate=0.1 '
        if info[3] == '2':
            remark += 'rate=0.2 '
        if info[3] == '5':
            remark += 'rate=0.5 '
    if info[4] == '0':
        if info[5] == '0':
            remark += 'Ollivier '
        if info[5] == '1':
            remark += 'Forman '
    accs = 0
    max_acc = -1
    min_acc = 100
    with open(filename, 'r') as f:
        num = 0
        for index, row in enumerate(f.readlines()):
            if index == 1:
                try:
                    row = row.replace('\'', '"')
                    dic = ast.literal_eval(row)
                    # if info[0]!= str(dic['curvature_activate_mode']) or info[1]!=str(dic['mask_mode']) or info[3]!=str(dic['mask_rate'])[-1] or info[4]!=str(dic['learnable_curvature']) or info[5]!=str(dic['curvature_method']) or dic['d_names'][0] != 'Cora':
                        # print(info)
                        # break
                        # print(row)
                except Exception:
                    print(1)
            error = row.find('IndexError')
            if error != -1:
                print(info)
                break
            index = row.find('ConvCurv Test')
            if index == -1:
                continue
            else:
                index += 15
                acc = float(row[index:])
                if acc>max_acc:
                    max_acc = acc
                elif acc<min_acc:
                    min_acc = acc
                accs += acc
                num +=1

    mean = round((max_acc+min_acc)/2,4)
    up = round(max_acc-mean,4)
    low = min_acc-mean
    print(f'{remark} {mean}±{up} {low}')

# for root, dirs, files in os.walk("logtest"):
#     for filename in files:
#         read_file(f'logtest/{filename}')

for i4 in range(3):
    for i1 in list1:
        for i2 in range(1, 3):
            if i2 == 2:
                for i3 in list2[1:]:
                    if i4 == 0:
                        for i5 in range(2):
                            filename = f'log0812/log{i1}{i2}.{i3}{i4}{i5}'

                            read_file(filename)
                    else:
                        filename = f'log0812/log{i1}{i2}.{i3}{i4}1'
                        read_file(filename)
            else:
                for i3 in list2:
                    if i4 == 0:
                        for i5 in range(2):
                            filename = f'log0812/log{i1}{i2}.{i3}{i4}{i5}'

                            read_file(filename)
                    else:
                        filename = f'log0812/log{i1}{i2}.{i3}{i4}1'
                        read_file(filename)
#!/usr/bin/env python

#################
# ETL6 & ETL7
#################

import numpy as np
import os

BYTES_IN_RECORD = 2052

width, height = 64, 63
out_size = 32
bpp = 4 # bits per pixel

def bits_to_image(bits):
    # add (8-bpp) bits for padding
    bits = bits.reshape(-1, bpp)
    pad = np.array([0]*bits.shape[0]*(8-bpp), dtype=np.uint8).reshape(-1, 8-bpp)
    bits = np.hstack([pad, bits])
    
    # float image
    data = np.packbits(bits).astype(np.float32)
    max = np.power(2, bpp)-1
    data = data / max
    # thresh
    data[data<0.5] = 0
    # append row & resize to 32x32
    data = np.append(data, np.zeros(width)).reshape([width, width])
    data = (data[::2, ::2] + data[1::2, ::2] + data[::2, 1::2] + data[1::2, 1::2])/4

    return data

def read_logical_record(record):
    image_bits = np.unpackbits(record[32:2048])
    return bits_to_image(image_bits)

def get_dictionary(files, chars):
    dic = {}
    for f in files:
        if not os.path.exists(f):
            print("Not found:", f)
            continue
        print("open:", f)
        rows = np.fromfile(f, dtype=np.uint8).reshape([-1, BYTES_IN_RECORD])

        for record in rows:
            char = record[2:4].tostring().decode('ascii')
            #print("char:", char)
            if char in chars:
                image = read_logical_record(record)
                if char not in dic:
                    dic[char] = image.ravel()
                else:
                    dic[char] = np.vstack([dic[char], image.ravel()])
    return dic



hiragana_files = [
    "ETL7/ETL7LC_1", "ETL7/ETL7LC_2", "ETL7/ETL7SC_1", "ETL7/ETL7SC_2"
]

hiragana_chars = [
     " I",
     "TO",
     "HE",
     "MO",
     "YU",
     "RI",
     " N",
     "- "
]

katakana_files = [
    "ETL6/ETL6C_01", "ETL6/ETL6C_02", "ETL6/ETL6C_03", "ETL6/ETL6C_04", 
    "ETL6/ETL6C_05", "ETL6/ETL6C_06", "ETL6/ETL6C_07", "ETL6/ETL6C_08", 
    "ETL6/ETL6C_09", "ETL6/ETL6C_10", "ETL6/ETL6C_11", "ETL6/ETL6C_12"
]

katakana_chars = [
    " A", " U", " O",
    "KA", "KI", "KU", "KE", "KO",
    "SI", "SU", "SO",
    "TA", "TI", "TU", "TE", 
    "NA", "NI", "NE", "NO",
    "HA", "HE",
    "MA", "MI", "ME", "MO",
    "RA", "RI", "RU", "RE", "RO",
    "WA", " N",
    "- "
]

target_chars = [ "SE", "NU", "HO", "WO" ]

hiragana_dic = get_dictionary(hiragana_files, hiragana_chars + target_chars)
katakana_dic = get_dictionary(katakana_files, katakana_chars + target_chars)

keyset = set(hiragana_dic.keys()) | set(katakana_dic.keys())

for k in keyset:
    if k in hiragana_dic and k in katakana_dic:
        v = np.vstack([hiragana_dic[k], katakana_dic[k]])
    elif k in hiragana_dic:
        v = hiragana_dic[k]
    elif k in katakana_dic:
        v = katakana_dic[k]
    else:
        continue
    
    images = v.reshape([-1, out_size, out_size, 1])
    print(k,":",images.shape[0])

    train_file_name = "data/train_"+k.strip()+".npy"
    test_file_name = "data/test_"+k.strip()+".npy"

    test_num = 10
    np.random.shuffle(images)
    np.save(train_file_name, images[test_num:])
    np.save(test_file_name, images[:test_num])
import hgtk

INPUT_DATA_FILE_NAME = "data/korean_spoiler.csv"
OUTPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted.csv"

except_char_list = ['ᴥ', '¿', 'º', '$', '§','°', 'ª','±', '¶']

with open(INPUT_DATA_FILE_NAME,"r", encoding='UTF8') as file:
    csv_data = []
    for line in file.readlines():
        csv_data.append(line.split(','))

with open(OUTPUT_DATA_FILE_NAME,"w", encoding='UTF8') as file_out:
    for i in csv_data:
        str_input = ""
        if i[1][0] == '0':
            str_input += "0 ,"
        else:
            str_input += "1 ,"
        hangul_split = hgtk.text.decompose(i[0])

        for i in except_char_list:
            hangul_split = hangul_split.replace(i,'')
        str_input += hangul_split + "\n"
        file_out.write(str_input)

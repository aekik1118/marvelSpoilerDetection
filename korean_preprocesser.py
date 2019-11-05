import korean_splitter as ks

INPUT_DATA_FILE_NAME = "data/korean_spoiler.csv"
OUTPUT_DATA_FILE_NAME = "data/korean_spoiler_splitted.csv"

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

        hangul_split = ks.KoreanSplitter.split_korean(i[0])
        str_input += hangul_split + "\n"
        file_out.write(str_input)

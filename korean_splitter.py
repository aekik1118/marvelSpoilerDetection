import hgtk


class KoreanSplitter:
    except_char_list = ['ᴥ', '¿', 'º', '$', '§', '°', 'ª', '±', '¶']

    def __init__(self):
        pass

    @staticmethod
    def split_korean(raw_str):
        hangul_split = hgtk.text.decompose(raw_str)

        for i in KoreanSplitter.except_char_list:
            hangul_split = hangul_split.replace(i,'')

        return hangul_split

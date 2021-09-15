
def get_script():
    SCRIPT = {}
    for line in open("src/regression/lang_data.txt"):
        cols = line.split()
        lang_code = cols[1]
        script = cols[3]
        SCRIPT[lang_code] = script
    return SCRIPT


def get_lang_family():
    SCRIPT = {}
    for line in open("src/regression/lang_data.txt"):
        cols = line.split()
        lang_code = cols[1]
        family = cols[4]
        if family == "IE:":
            family += cols[5]
        SCRIPT[lang_code] = family
    print(SCRIPT)
    return SCRIPT


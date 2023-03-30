# %%
import glob
import re
import os
# %%
RES_PATH = "FWD_data/addresses"
LOG_PATH = "log/addresses_crawler.log"
FWD_OCR_PATH = '/home/sds/hungbnt/ocr/result/FWD_full/'


def get_fwd_files():
    return glob.glob(f"{FWD_OCR_PATH}/*.txt")


def split_address(text):
    return re.search(r"Tổ chức\)?(.*)", text).group(1)


def get_address_from_FWD_form(file):
    with open(file, 'r') as f:
        lines = f.read().splitlines()
        res = ""
        try:
            for i, line in enumerate(lines):
                if 'Địa chỉ thường trú' in line or 'trụ sở chính' in line:
                    res += split_address(line)
                    if "Địa chỉ" not in lines[i + 1] and "gửi thư" not in lines[i + 1]:
                        res += lines[i + 1]
            return res
        except:
            print('[ERROR] : ', file)
            return ""


def main():
    lfiles = get_fwd_files()
    for file in lfiles:
        file_name = file.split(FWD_OCR_PATH)[-1]
        file_save = os.path.join(RES_PATH, file_name)
        res = get_address_from_FWD_form(file)
        if res == "":
            with open(LOG_PATH, 'a') as f:
                f.write(file_save)
                f.write('\n')
        with open(file_save, 'w') as f:
            f.write(res)


# %%
if __name__ == "__main__":
    main()
# %%
# a = 'h. Địa chỉ thường trú (hoặc địa chỉ tru sở chính của Tổ chức'

# print(split_address(a))
# %%

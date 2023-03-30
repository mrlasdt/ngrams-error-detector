#%%
from tools.utils import read_data, write_data
from runner import Runner
import glob
from tqdm import tqdm
import os
import unicodedata
predictor =Runner(istrain=False, mode='bothward')
log_path = 'inference.log'

# #%%
# def log_to_file(i, text, pred):
#     with open(log_path, 'a') as f:
#         f.write(str(i) + ' : ' +str(text))
#         f.write(str(pred) +'\n')

# file_path = 'corpus/danhba_org_drop_duplicated.txt'
# # file_path = 'corpus/danhba_org/ha_noi/quan_cau_giay/phuong_dich_vong.txt'
# texts = read_data(file_path)
# for i, text in enumerate(tqdm(texts)):
#     # text = 'quận cầu giấy'
#     pred = predictor.inference_address(text)
#     if sum(pred)!=0:
#         log_to_file(i, text, pred)
#             # print('[INFO]: ', file_path)
#             # print('[INFO]: ', text)
#             # print('[INFO]: ', pred)
#             # break
#         break
#     break
# %%
text = "Aa Điền X Cộng Hòa, H. Nam Sách, T. Hai Dương"
print(predictor.inference_address(text))

# %%

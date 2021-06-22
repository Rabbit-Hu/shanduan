import ctext
import os
import torch

ctext.setapikey("aas2019")
ctext.setremap("gb")

book_list = ["shiji", "han-shu", "hou-han-shu", "sanguozhi"]
# book_name = "shiji"

# data = ctext.gettextasstring("ctp:" + book_name)
# torch.save(data, "./data/" + book_name + "_text.pth")
# data = torch.load("./data/" + book_name + "_text.pth")
os.makedirs("./data/txt_files", exist_ok=True)
for book_name in book_list:
    data = ctext.gettextasstring("ctp:" + book_name)
    with open("./data/" + book_name + ".txt", "w") as f:
        f.write(data)

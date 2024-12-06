import os
from aiki.aiki import AIKI


a_k_ = AIKI(db_name = "flick")

a_k_.index("我上周出去遛狗了")
image_path = os.path.abspath("resource/source/imgs/外滩小巷.jpg")
a_k_.index(image_path)

print(a_k_.retrieve("我上周出去遛狗了么", num=2))
print(a_k_.retrieve("几个人在街上，有些人正在使用手机，另外一些人在骑自行车", num=2))

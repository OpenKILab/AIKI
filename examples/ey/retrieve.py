from aiki.aiki import AIKI

ak = AIKI(db_path="/mnt/hwfile/kilab/leishanzhe/db/ey/")

result = ak.retrieve("经济", num=10)
print(result)
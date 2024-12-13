# # import os

# # from aiki.database.chroma import ChromaDB

# # db_path="/mnt/hwfile/kilab/leishanzhe/db/wiki/"
# # chroma_db = ChromaDB(collection_name=f"index", persist_directory=os.path.join(db_path, "index"))

# # print("===============")
# # print(chroma_db.count())

# from PIL import Image
# import requests

# from transformers import CLIPProcessor, CLIPModel

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
# print(outputs)


# import time
# import requests
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

# # 加载预训练的CLIP模型和处理器
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # 加载图像并进行预处理
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# # inputs = processor(images=image, return_tensors="pt")
# # 计算处理时间
# start_time = time.time()

# inputs = processor(text=["海滩上美丽的日落"], return_tensors="pt")
# # 提取图像特征
# with torch.no_grad():
#     # image_features = model.get_image_features(**inputs)
#     image_features = model.get_text_features(**inputs)

# end_time = time.time()
# processing_time = end_time - start_time
# print(f"处理时间: {processing_time:.4f} 秒")

# 打印图像特征
# print(image_features)
# print(image_features.tolist()[0].tolist())

'''
tensor([[-1.0570e-01,  1.3791e-01, -2.9611e-01,  2.1249e-02, -6.4070e-02,
         -1.6862e-01, -1.3514e-01, -2.4488e-03,  4.7376e-01, -1.7627e-01,
          2.4440e-01, -3.7972e-01,  4.8326e-02, -1.3981e-01, -3.4045e-01,
         -1.2676e-01, -2.3266e-01, -2.9760e-01,  1.7886e-01,  4.9608e-02,
         -1.3074e+00, -3.2436e-02,  4.2144e-01, -3.3363e-01, -4.7374e-02,
          2.9804e-01,  2.3910e-01, -1.8429e-01,  1.5669e-01, -4.8810e-02,
         -7.7587e-02,  2.5888e-01, -7.3554e-02,  1.7692e-01, -5.7918e-01,
         -4.9198e-03,  2.7906e-01, -2.8707e-01,  1.9080e-01,  3.2633e-01,
         -1.0051e-01, -3.4647e-01,  7.1661e-03, -1.4746e-01, -1.9010e-01,
          5.7498e-04,  5.2022e-01,  1.5117e-01, -9.1645e-02,  1.7595e-01,
          1.6325e-01,  2.5157e-01,  1.1609e-01, -5.3788e-02,  2.2871e-01,
          1.8608e-01,  2.5763e-01,  6.0314e-01, -2.5823e-01, -1.6781e-01,
          4.2385e-01, -1.3951e-01, -6.3743e-02,  3.5568e-01, -7.4244e-02,
         -2.7096e-01,  2.2417e-01,  1.0962e+00,  1.9659e-01,  7.8918e-02,
          2.0099e-01, -1.0657e-01,  2.7643e-01,  2.7661e-01,  4.9342e-01,
         -1.6944e-02,  6.3948e-02, -3.4031e-01, -5.2761e-02, -1.3829e-01,
         -1.8781e-01, -6.7633e-01, -4.5840e-01, -1.5787e-02, -5.5268e-01,
          2.6263e-01,  4.3102e-01, -5.3736e-01,  2.9391e-01,  9.3340e-02,
          1.7963e-01,  2.7745e-02, -7.3421e+00,  3.2605e-01, -2.1535e-01,
          6.7097e-02,  1.7348e-01, -4.2522e-01, -9.2658e-01,  1.2663e+00,
         -2.9132e-02,  6.1214e-02, -3.0542e-01, -7.5486e-03,  7.2453e-01,
          2.5023e-01,  1.2300e+00,  2.6843e-01, -1.8051e-01, -4.7297e-01,
         -1.5611e-01, -1.0561e+00,  6.2212e-02,  1.9993e-01,  1.8817e-01,
          5.9712e-02,  8.6168e-02,  1.1679e-01,  1.7643e-01, -1.0635e-01,
          1.6236e-01, -2.1022e-01, -1.8189e-01,  3.9629e-01, -1.6404e-01,
         -9.2917e-02, -6.0283e-02,  5.1839e-01, -1.2987e-01,  2.8777e-02,
         -5.2972e-02,  7.8340e-02, -1.9674e-01,  9.4795e-01, -4.1044e-01,
         -3.8089e-01,  1.4848e-01, -4.8702e-01, -4.1070e-01, -1.2976e-01,
          1.5802e-01, -3.7492e-01,  3.5367e-01,  4.1437e-01, -1.5291e-01,
          2.8547e-01,  3.2227e-01, -5.6265e-01,  8.8450e-02,  2.4728e-01,
         -4.8397e-01,  1.8799e-01, -3.0984e-01, -2.2551e-01,  1.9980e-02,
         -5.9366e-02, -3.3540e-01, -4.7573e-01,  1.1815e-01, -4.0058e-01,
         -2.7257e-02, -2.6316e-01,  1.1217e-01,  2.2248e-01,  3.2559e-04,
         -3.8470e-01, -5.2015e-01,  5.2815e-01,  1.4545e-01,  3.0454e-01,
          2.2329e-01,  6.8722e-01,  3.0521e-01, -2.2130e-01, -2.1167e-01,
          1.8806e-01,  2.2055e-02,  5.4690e-02,  6.9693e-02,  2.5776e-01,
         -3.7218e-01,  2.3236e-01, -1.2269e-01, -3.6241e-01, -5.8699e-02,
          1.3109e-01,  1.6811e-01, -3.7950e-01,  3.2862e-01, -1.7761e-01,
          3.4696e-01, -2.8875e-01, -5.4294e-01, -1.3271e-01, -4.8734e-01,
         -3.3036e-02, -2.6275e-01, -3.4988e-01, -1.1122e+00, -4.2773e-01,
          4.3965e-02,  3.6076e-01,  1.3401e-01,  1.8443e-01, -1.0153e-01,
         -1.2240e-02,  3.2831e-01,  6.7399e-02, -3.0257e-01,  5.6250e-01,
         -4.1210e-01,  5.1786e-01, -3.5435e-02,  3.3271e-01, -4.2546e-01,
         -3.5230e-01, -2.9020e-03, -7.1604e-01,  8.2598e-01, -3.9524e-01,
          3.6184e-03,  1.5516e-01, -1.9975e-01,  1.6137e-01, -4.9928e-02,
          3.6856e-01,  1.4779e-01, -6.9431e-02,  2.4152e-02, -1.9426e-01,
         -1.5363e-01,  5.1793e-02, -3.3300e-01,  7.6162e-01,  2.7636e-01,
         -5.3359e-01, -3.5004e-01,  1.3252e-01,  1.4886e-01, -1.8810e-01,
          3.5507e-01, -1.8194e-01, -1.2301e-01, -2.9974e-02, -3.2487e-01,
          6.2948e-01,  1.2428e-01, -4.6405e-02, -1.0913e-01,  7.0813e-01,
          1.4804e-02,  2.7664e-01, -1.6041e-01, -7.4244e-02,  1.9206e-01,
          4.8417e-02, -4.0808e-02, -2.7543e-01, -6.3991e-01, -4.4215e-02,
         -8.5915e-02,  3.3263e-02, -3.5333e-03,  4.5338e-01,  3.2682e-01,
         -1.0774e-01, -5.3721e-01, -4.2815e-01, -2.8198e-01,  6.1129e-02,
         -2.4625e-01, -2.0745e-01,  3.1999e-01, -1.3205e-01, -2.2901e-01,
         -1.8604e-01, -1.7491e-02,  1.7489e-01, -4.3999e-01,  1.0667e-01,
          3.4010e-01,  2.5526e-01, -3.8551e-01, -6.2775e-02, -2.2571e-01,
          7.5087e-02,  1.7720e-02, -1.3168e-02,  4.8330e-01, -1.9816e-01,
         -1.4375e-01,  1.7781e-01,  2.7621e-01,  3.0707e-01,  1.8930e-01,
          1.9520e-01,  2.6625e-01, -5.7480e-01,  7.9328e-03, -1.5115e-01,
          2.3183e-01, -1.4169e-01, -1.1889e-01,  4.3253e-01, -9.8077e-02,
          3.0851e-01, -6.7046e-02,  3.7698e-01,  8.8926e-02, -1.8221e-01,
         -5.8740e-01,  2.7006e-01,  9.4688e-01, -6.2511e-02, -1.8787e-01,
          2.5662e-01,  1.5217e-01,  4.8820e-01,  1.8438e-01,  2.5828e-01,
         -1.9514e-01,  1.6153e+00, -5.8381e-01, -3.6166e-02, -2.4056e-01,
         -1.3784e-01, -9.4208e-02,  5.7140e-01,  1.9949e-01,  2.6506e-02,
          1.4178e-02, -1.5593e-01, -1.7336e-01, -1.4924e-01,  3.4075e-02,
          5.3849e-01,  7.3388e-02,  1.1542e-01, -1.5296e-02, -2.2717e-02,
         -1.9322e-02, -5.7205e-02,  2.1121e-01,  7.7404e-02,  2.0938e-01,
         -4.9050e-01, -1.2306e-01,  1.6134e-01, -8.3677e-02,  3.4774e-02,
          1.1503e-01, -2.2666e-01,  5.5621e-01, -4.0450e-01, -2.0487e-01,
         -1.0600e-01, -2.7015e-01,  2.6361e-01,  2.0263e-01,  3.7978e-01,
          1.5335e-01,  1.2605e-01,  5.0422e-04, -1.3195e-01, -4.7008e-01,
          1.9779e-01,  2.9090e-01, -5.4812e-01,  5.8499e-01, -1.0590e-01,
          4.7545e-01,  5.0500e-01, -2.1749e-01, -5.9405e-01,  3.2484e-01,
         -1.5170e-01,  1.8060e+00,  7.1680e-02, -4.2355e-01, -7.4224e-02,
          6.7890e-02, -5.1898e-01, -1.6756e-01,  4.2614e-01, -1.2730e-01,
          1.6015e-02, -1.1366e-01, -4.3086e-01,  1.2056e-01, -8.4343e-01,
         -3.2650e-01, -4.5064e-01,  1.1179e-01, -2.6111e-01,  1.9197e-01,
          1.2052e-01,  4.5279e-01, -3.2545e-01, -3.5914e-01, -2.7329e-01,
          3.7605e-02,  2.1050e-01,  4.3315e-01,  2.8789e-01,  9.5152e-02,
         -3.6645e-02, -2.0286e-01, -4.5810e-01,  6.2415e-01,  3.4840e-02,
          5.0373e-01,  7.9766e-01, -1.8451e-01,  2.6256e-01,  5.5923e-04,
         -6.5547e-01, -3.3765e-01, -6.3588e-02, -4.0205e-02, -3.6788e-01,
         -5.8009e-01,  3.2927e-01,  2.0086e-01,  4.2279e-01,  1.2953e-01,
          2.4375e-03,  1.1159e-01, -3.7128e-01,  6.6182e-01,  2.8008e-01,
         -3.2797e-01,  7.9389e-01, -5.1525e-01,  2.6869e-01, -4.8718e-02,
         -5.8017e-01, -3.9886e-02,  4.2184e-01,  2.9164e-01,  1.2954e-02,
         -1.1650e-01, -2.6768e-01,  6.7934e-02, -1.1775e-01,  3.4765e-01,
         -3.4615e-02,  5.2753e-02, -2.9523e-01, -1.9536e-01, -4.1873e-01,
         -5.5842e-01, -2.5225e-01, -5.3106e-01,  5.3523e-01, -1.2160e-01,
         -3.0841e-01,  3.7261e-01,  4.4894e-01,  1.3557e-01,  4.5114e-01,
         -3.2811e-01, -3.9681e-01,  8.4930e-02, -2.9096e-01,  4.1583e-01,
          1.4222e-01, -2.9784e-01, -2.1871e-01, -1.9392e-01, -4.0934e-01,
          4.2811e-01, -5.0897e-01, -4.1473e-01, -1.0547e-01, -6.7014e-01,
         -4.8981e-01, -3.7619e-01, -1.3544e-01, -1.2337e-01, -7.2206e-01,
         -6.2175e-02, -9.9811e-02,  5.5191e-02,  7.3064e-02,  2.4990e-01,
          1.5880e-01, -2.0090e-01,  1.1332e-01, -2.5104e-01, -1.5370e-01,
         -4.3505e-02, -8.5118e-02, -1.9082e-01, -4.1292e-01,  6.2387e-01,
          1.9057e-01,  7.9786e-02,  3.0655e-01,  1.2087e-01,  1.3618e-01,
         -1.6667e-01,  5.7030e-01, -3.0686e-01, -1.7335e-01,  7.3685e-02,
         -2.2815e-01,  1.3495e-01, -4.4910e-01,  1.2777e-01,  8.6682e-01,
         -1.4580e-02,  2.5632e-01]])
'''


# !pip install sentence-transformers einops timm pillow
import random
from sentence_transformers import SentenceTransformer
import time
# Choose a matryoshka dimension
truncate_dim = 512

# Initialize the model
model = SentenceTransformer(
    'jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=truncate_dim, device='cuda'
)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
    'num_texts: 要生成的文本数量',
    'max_length: 每个文本的最大长度',
    'nerate_random_text(min_length=5, max_length=20',
    '可以指定文本的最小和最大长度',
    '如果你想要更改字符集或其他参数，可以：',
    '1. 使用了所有ASCII字母（大小写）、数字、标点符号和空格',
    '每种语言都有多个相似但不完全相同的表达方式',
    '扩充每种语言的模板句子',
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg']
while True:
    # Encode text and images
    # text_embeddings = model.encode(sentences, normalize_embeddings=True)
    start_time = time.time()
    input = random.choice(sentences)
    print(f"input: {input}")
    image_embeddings = model.encode(
        [input], normalize_embeddings=True
    )  # also accepts PIL.Image.Image, local filenames, dataURI

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"jina编码处理时间: {processing_time:.4f} 秒")
    time.sleep(.5)

# # Encode query text
# query = 'beautiful sunset over the beach' # English
# query_embeddings = model.encode(
#     query, prompt_name='retrieval.query', normalize_embeddings=True
# )

# print(image_embeddings)
# print(image_embeddings[0].tolist())

# from sentence_transformers import SentenceTransformer

# start_time = time.time()  # Add timing start

# sentences_1 = ["海滩上美丽的日落"]
# model = SentenceTransformer('DMetaSoul/Dmeta-embedding')
# embeddings_1 = model.encode(sentences_1)

# end_time = time.time()  # Add timing end
# processing_time = end_time - start_time
# print(f"Dmeta编码处理时间: {processing_time:.4f} 秒")
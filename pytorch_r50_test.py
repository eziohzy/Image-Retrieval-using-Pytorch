import torch 
import torchvision
# Tag :图像预处理包,包含了很多种对图像数据进行变换的函数
import torchvision.transforms as transforms
import torch.nn as nn
# Tag 
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from cifar10_models.vgg import *
# from cifar10_models import densenet161 #vgg16_bn
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import time
from datasketch import MinHash, MinHashLSH
import os
from torchvision import datasets

# path="/content/drive/MyDrive/PyTorch_CIFAR10-master/cifar10_models/state_dicts/resnet50.pt"
output_dim = 100
LOAD_FEATURE_DICT=True
LOAD_QUERY_DICT=True


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
val_transforms = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


batch_size=1
# val_dir = './VGGDataSet/val'
val_dir = '/content/drive/MyDrive/gallery'
val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
testloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

test_dir = '/content/drive/MyDrive/query'

test_datasets = datasets.ImageFolder(test_dir, transform=val_transforms)
queryloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)


# load Net model
# 下载官方权重
# net = resnet50(pretrained=True)
# net = vgg16_bn(pretrained=True)
net = vgg16(pretrained=True)
#load weight
# net.load_state_dict(torch.load(path),strict=False)
# net2=torch.load(path):
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
summary(net,(3, 224, 224))

DATASET_VEC_PATH='dataset_vec.npy'
DATASET_IMG_PATH='dataset_img.npy'
DATASET_LABEL_PATH='dataset_label.npy'
QUERY_VEC_PATH='QUERY_vec.npy'
QUERY_IMG_PATH='QUERY_img.npy'
QUERY_LABEL_PATH='QUERY_label.npy'

from PIL import Image
from numpy import linalg as LA
image_PIL = Image.open('/content/drive/MyDrive/gallery/accordion/image_0001.jpg')
image_tensor = train_transforms(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 没有这句话会报错
with torch.no_grad():
    image_tensor = image_tensor.to(device)
    outputs = net(image_tensor)
    output_np=outputs.cpu().numpy()
    print(output_np.shape)
    # print("output_np",output_np)
    output_np= np.squeeze(output_np)
    # print("after squeeze", output_np)
    print("first image tensor", output_np/LA.norm(output_np))
# testimg
correct = 0
total = 0
# if(LOAD_FEATURE_DICT):
if os.path.exists(DATASET_VEC_PATH):
    print("dataset file exist")
# save the feature_dict
    dataset_vec = np.load(DATASET_VEC_PATH,allow_pickle='TRUE')
    dataset_img = np.load(DATASET_IMG_PATH,allow_pickle='TRUE')
    dataset_label = np.load(DATASET_LABEL_PATH,allow_pickle='TRUE')
else:
    # feature_dict={}
    dataset_vec=[]
    dataset_label=[]
    dataset_img=[]
    with torch.no_grad():
        for data in testloader:
            net.eval()
            images, labels = data
            #Tag: 、it's  already convert to tensor
            # print(images)

            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            output_np=outputs.cpu().numpy()
            # print(output_np.shape)
            output_np= np.squeeze(output_np)
            # feature_dict.update({images:output_np})
            # feature_dict.update({labels:output_np})
            dataset_img.append(images)
            dataset_label.append(labels)
            dataset_vec.append(output_np)
            # for i in range(output_np.shape[0]):
            #   feature_list.append(output_np[i,:])
    print(len(dataset_vec))
    np.save(DATASET_VEC_PATH, dataset_vec)
    np.save(DATASET_IMG_PATH, dataset_img)
    np.save(DATASET_LABEL_PATH, dataset_label)

if os.path.exists(QUERY_VEC_PATH):
# if(LOAD_QUERY_DICT):
# save the feature_dict
    print("query file exist")
    query_vec = np.load(QUERY_VEC_PATH,allow_pickle='TRUE')
    query_img = np.load(QUERY_IMG_PATH,allow_pickle='TRUE')
    query_label = np.load(QUERY_LABEL_PATH,allow_pickle='TRUE')
else:
    query_vec=[]
    query_label=[]
    query_img=[]
    with torch.no_grad():
        for data in queryloader:
    # data = queryloader[0]
            net.eval()
            images, labels = data
            # 已经被转为tensor
            # print("data", data)
            # print("images", images)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # print(outputs)
            output_np=outputs.cpu().numpy()
            output_np= np.squeeze(output_np)
            # query_dict.update({images:output_np})
            # query_dict.update({labels:output_np})
            query_img.append(images)
            query_label.append(labels)
            query_vec.append(output_np)
            # for i in range(newX.shape[0]):
            # query_list.append(output_np)
        # print(query_list)
    print(len(query_vec))
    np.save(QUERY_VEC_PATH, query_vec)
    np.save(QUERY_IMG_PATH, query_img)
    np.save(QUERY_LABEL_PATH, query_label)

def imshow(img, label):
    # img=img/2+0.5
    npimg=img.cpu().numpy()

    label_=label.cpu().numpy()
    # print(label_)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(str(label_))
    # plt.show()
def axshow(ax, img, label):
    # img=img/2+0.5
    npimg=img.cpu().numpy()
    label_=label.cpu().numpy()
    ax.imshow(np.transpose(npimg,(1,2,0)))
    ax.set_title(str(label_))
INDEX=0
query_image = dataset_img[INDEX]
imshow(query_image[0], dataset_label[INDEX])
plt.savefig("test")






# calculate the euclident distance. and return  the top 5
def calc_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1-vec2)))
# how to show the img

# Calculate the mAP, and rank1-accu
# for key in (feature_dict).items():
# for kv in a.items():
AP=0
right_predict=0
right_topk=0
CAN_NUM=5
K=1
time_cost=0
topK_satisfy=False
OUT_IMG=True
# print("dataset_vec[0]", dataset_vec[0])

for i in range(len(query_vec)):
    start = time.clock()
    query_img_label=query_label[i]
    query_img_vec = query_vec[i]
    # print("query_img_vec",i,query_img_vec)
    candidates=[]
    for k in range(len(dataset_vec)):
        # Eucli
        # print("dataset_vec[k]",k,dataset_vec[k])
        # distance = calc_distance(query_img_vec, dataset_vec[k])
        # Cos distance?
        query_norm = query_img_vec/LA.norm(query_img_vec)
        dataset_norm = dataset_vec[k]/LA.norm(dataset_vec[k])
        distance= np.dot(query_norm, dataset_norm.T)
      # incase that the calc distance is the same
        candidates.append(distance)
      # print(distance)
    # sort
  # change into list?'list' object
    # can_list = sorted(candidates.items(), key=lambda x: x[1])
    # can_list = candidates.sort()
    # print("candidates",i,candidates)
    can_indexs=[]
    # 小到大
    # can_indexs = sorted(range(len(candidates)), key=lambda k: candidates[k])
    # 大到小
    can_indexs = sorted(range(len(candidates)), key=lambda k: candidates[k], reverse=True)
    # print('元素索引序列：', can_indexs)
    for test_num in range(CAN_NUM):
       print('top 元素索引序列：', test_num,can_indexs[test_num], candidates[can_indexs[test_num]]) 
  # print("len(candidates)",len(candidates))
    # can_list =[(key, kv)for key, kv in candidates.items()]


    end = time.clock()
    time_cost+=(end-start)
    if(OUT_IMG):
        fig,(ax1,ax2,ax3,ax4,ax5,ax6)=plt.subplots(1,6)
        # print("i",i)
        axshow(ax1, query_img[i][0], query_label[i])
        ax_list=[ax2,ax3,ax4,ax5,ax6]
        for j in range(CAN_NUM):    
            # if(can_list[j][0]==query_img_label):
            axshow(ax_list[j], dataset_img[can_indexs[j]][0], dataset_label[can_indexs[j]])
        plt.savefig("query")
        OUT_IMG=False

    # print("time_cost",time_cost)
    # first_5_can_list=can_indexs[:CAN_NUM]
    print("query_label[i]", query_label[i])
    for j in range(CAN_NUM):    
        # if(can_list[j][0]==query_img_label):
        print("dataset_label[can_indexs[j]]", dataset_label[can_indexs[j]]), 
        if(dataset_label[can_indexs[j]]==query_label[i]):
            if(j<K):
                topK_satisfy=True
            right_predict+=1
    if(topK_satisfy):
        right_topk+=1
        topK_satisfy=False
    print("i, right_predict, right_topk", i, right_predict, right_topk)
    AP+=right_predict/CAN_NUM
    
    right_predict=0
    mAP=AP/(i+1) 
    print("AP,mAP",AP,mAP) 
topK=right_topk/len(query_vec)
ave_time_cost=time_cost/len(query_vec)
print("ave_time_cost",ave_time_cost)
print("mAP",mAP)
print("topK",topK)




'''
# construct the Hash table.
dataset_m_list=[]
start =time.clock()
for i in range(len(dataset_vec)):
# for i in range(3):
    m=MinHash(num_perm=128)
    for j in dataset_vec[i]:
        m.update(str(j).encode('utf8'))
    dataset_m_list.append(m)

query_m_list=[]
for i in range(len(query_vec)):
# for i in range(3):
    m=MinHash(num_perm=128)
    for j in query_vec[i]:
        m.update(str(j).encode('utf8'))
    query_m_list.append(m)

# print(m_list)
#the perform search 

lsh = MinHashLSH(threshold=0.5, num_perm=128)
i=0
for item in dataset_m_list:
    lsh.insert((i), item)
    i+=1
end=time.clock()
print("build hash table takes ",(end-start))

start =time.clock()
result=lsh.query(query_m_list[0])
end=time.clock()
print("one query takes ",(end-start))
print(result)
for i in result:
    print(i)

fig,(ax1,ax2)=plt.subplots(1,2)
# print("i",i)
i=0
axshow(ax1, query_img[i][0], query_label[i])
ax_list=[ax2]
j=0
axshow(ax_list[j], dataset_img[result[j]][0], dataset_label[result[j]])
plt.savefig("lsh_query")'''
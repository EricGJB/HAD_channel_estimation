import numpy as np
np.random.seed(2021)
from scipy import io
from matplotlib import pyplot as plt

# system parameters
Nt = 64
Nr = 4
Lp = 20
AS = 5

# generate dataset
data_num = 50000
H_list = np.zeros((data_num,Nt,Nr))+1j*np.zeros((data_num,Nt,Nr))
angular_channel_list = np.zeros((data_num,Nt,Nr))+1j*np.zeros((data_num,Nt,Nr))
Gains_list = np.sqrt(1/2)*(np.random.randn(data_num,Lp)+1j*np.random.randn(data_num,Lp))
# 10-15 region, but train with data from start-1 to end+1 if GPS error is 1 degree at max
mean_angle_list = np.random.uniform(10-1,15+1,data_num)

def DFT_matrix(N):
    F = np.zeros((N,N))+1j*np.zeros((N,N))
    # DFT矩阵每一行对应将角度的sin值进行N等分
    for i in range(-int(N/2),int(N/2)):
        for k in range(N): 
            F[i+int(N/2),k] = np.exp(-1j*2*np.pi*(i+1/2)*k/N)/np.sqrt(N)
    return F
F = DFT_matrix(Nt)

for i in range(data_num):
    if i%1000==0:
        print('%d/%d'%(i,data_num))
    mean_angle = mean_angle_list[i]
    AoAs = np.random.uniform(mean_angle-AS,mean_angle+AS,Lp)/180*np.pi
    AoDs = np.random.uniform(-1/2,1/2,Lp)*np.pi
    for lp in range(Lp):
        steering_vector_BS = np.expand_dims(np.exp(1j*2*np.pi*1/2*np.arange(Nt)*np.sin(AoAs[lp])),-1)
        steering_vector_user = np.expand_dims(np.exp(1j*2*np.pi*1/2*np.arange(Nr)*np.sin(AoDs[lp])),0)
        H_list[i] = H_list[i] + Gains_list[i,lp]*steering_vector_BS.dot(steering_vector_user)
    angular_channel_list[i] = F.dot(H_list[i])
    
H_list = H_list/np.sqrt(Lp)
angular_channel_list = angular_channel_list/np.sqrt(Lp)
print(H_list.shape)
print(angular_channel_list.shape)

# plt.plot(np.abs(angular_channel_list[3,0,:,0]))
# plt.plot(np.abs(angular_channel_list[3,1,:,1]))
# plt.plot(np.abs(angular_channel_list[4,1,:,0]))

# directly save the angular domain channel
io.savemat('./data/paper_10_15_64_4.mat',{'channels':angular_channel_list})
print('Data saved!')

import numpy as np
np.random.seed(2021)
from scipy import io

# system parameters
Nt = 128
Lp = 10
AS = 5

# generate dataset
data_num = 50000
H_list = np.zeros((data_num,Nt))+1j*np.zeros((data_num,Nt))
#mean_angle_list = np.random.uniform(0,360,data_num)
mean_angle_list = np.random.uniform(15,20,data_num)
Gains_list = np.sqrt(1/2)*(np.random.randn(data_num,Lp)+1j*np.random.randn(data_num,Lp))
H_list = np.zeros((data_num,Nt))+1j*np.zeros((data_num,Nt))

for i in range(data_num):   
    if i%1000==0:
        print('%d/%d'%(i,data_num))
#    DoAs = np.random.uniform(-1/2,1/2,Lp)*np.pi
    mean_angle = mean_angle_list[i]
    DoAs = np.random.uniform(mean_angle-AS,mean_angle+AS,Lp)/180*np.pi
    for lp in range(Lp):
        H_list[i] = H_list[i] + Gains_list[i,lp]*np.exp(1j*2*np.pi*1/2*np.arange(Nt)*np.sin(DoAs[lp]))    

H_list = H_list/np.sqrt(Lp)
print(H_list.shape)

io.savemat('./data/H_list_%d.mat'%Nt,{'H_list':H_list})
print('Data saved!')




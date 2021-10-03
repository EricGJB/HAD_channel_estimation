#%%
import numpy as np
np.random.seed(2021)
import tensorflow as tf
tf.random.set_seed(2021)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GaussianNoise,GlobalAvgPool1D,Multiply,Input,Dense,Conv1D,Reshape,Permute,Lambda,Flatten,Add,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

from scipy import io
from matplotlib import pyplot as plt
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)



#%%
def get_received_signal(x_real,x_imag,fai_real,fai_imag):
    fai_real_x_real = Lambda(lambda x:tf.matmul(x[0],x[1]))([fai_real,x_real])
    fai_real_x_imag = Lambda(lambda x:tf.matmul(x[0],x[1]))([fai_real,x_imag])
    fai_imag_x_real = Lambda(lambda x:tf.matmul(x[0],x[1]))([fai_imag,x_real])
    fai_imag_x_imag = Lambda(lambda x:tf.matmul(x[0],x[1]))([fai_imag,x_imag])
    fai_imag_x_imag = Lambda(lambda x:-x)(fai_imag_x_imag)

    y_real = Add()([fai_real_x_real,fai_imag_x_imag])
    y_imag = Add()([fai_real_x_imag,fai_imag_x_real])
    y = Lambda(lambda x:tf.concat([x[0],x[1]],axis=-1))([y_real,y_imag])
    return y


class B_Layer(tf.keras.layers.Layer):
  def __init__(self,num_inputs):
    super(B_Layer, self).__init__()
    self.num_inputs = num_inputs
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",shape=[self.num_inputs,self.num_inputs,2],trainable=False)
  def call(self, input):
    input_real = input[:,:,0:1]
    input_imag = input[:,:,1:2]
    batch_zeros = input_real-input_real
    batch_zeros = tf.expand_dims(tf.expand_dims(batch_zeros[:,0,0],-1),-1)
    batch_zeros = tf.tile(batch_zeros,(1,self.num_inputs,self.num_inputs))
    B_real = self.kernel[:,:,0]+batch_zeros
    B_imag = self.kernel[:,:,1]+batch_zeros
    RR = tf.matmul(B_real,input_real)
    RI = tf.matmul(B_real,input_imag)
    IR = tf.matmul(B_imag,input_real)
    II = tf.matmul(B_imag,input_imag)
    output_real = RR-II
    output_imag = RI+IR
    return output_real,output_imag


class W_Layer(tf.keras.layers.Layer):
  def __init__(self,num_inputs,num_outputs,trainable):
    super(W_Layer, self).__init__()
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
    self.trainable = trainable
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",shape=[self.num_outputs,self.num_inputs],trainable=self.trainable)
  def call(self, input):
    input_real = input[:,:,0:1]
    input_imag = input[:,:,1:2]
    batch_zeros = input_real-input_real
    batch_zeros = tf.expand_dims(tf.expand_dims(batch_zeros[:,0,0],-1),-1)
    batch_zeros = tf.tile(batch_zeros,(1,self.num_outputs,self.num_inputs))
    theta_real = tf.cos(self.kernel)+batch_zeros
    theta_imag = tf.sin(self.kernel)+batch_zeros
    RR = tf.matmul(theta_real,input_real)
    RI = tf.matmul(theta_real,input_imag)
    IR = tf.matmul(theta_imag,input_real)
    II = tf.matmul(theta_imag,input_imag)
    output_real = RR-II
    output_imag = RI+IR
    outputs = tf.concat([output_real,output_imag],axis=-1)
    return outputs


class WBB_Layer(tf.keras.layers.Layer):
  def __init__(self,num_inputs,trainable):
    super(WBB_Layer, self).__init__()
    self.num_inputs = num_inputs
    self.trainable = trainable
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",shape=[self.num_inputs,self.num_inputs,2],trainable=self.trainable)
  def call(self, input):
    input_real = input[:,:,0:1]
    input_imag = input[:,:,1:2]
    batch_zeros = input_real-input_real
    batch_zeros = tf.expand_dims(tf.expand_dims(batch_zeros[:,0,0],-1),-1)
    batch_zeros = tf.tile(batch_zeros,(1,self.num_inputs,self.num_inputs))
    theta_real = self.kernel[:,:,0]+batch_zeros
    theta_imag = self.kernel[:,:,1]+batch_zeros
    RR = tf.matmul(theta_real,input_real)
    RI = tf.matmul(theta_real,input_imag)
    IR = tf.matmul(theta_imag,input_real)
    II = tf.matmul(theta_imag,input_imag)
    output_real = RR-II
    output_imag = RI+IR
    outputs = tf.concat([output_real,output_imag],axis=-1)
    return outputs


class fai_Layer(tf.keras.layers.Layer):
  def __init__(self,num_inputs,num_outputs):
    super(fai_Layer, self).__init__()
    self.num_inputs = num_inputs
    self.num_outputs = num_outputs
  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",shape=[self.num_outputs,self.num_inputs,2],trainable=True)
  def call(self, input):
    input_real = input[:,:,0:1]
    input_imag = input[:,:,1:2]
    batch_zeros = input_real-input_real
    batch_zeros = tf.expand_dims(tf.expand_dims(batch_zeros[:,0,0],-1),-1)
    batch_zeros = tf.tile(batch_zeros,(1,self.num_outputs,self.num_inputs))
    theta_real = self.kernel[:,:,0]+batch_zeros
    theta_imag = self.kernel[:,:,1]+batch_zeros
    RR = tf.matmul(theta_real,input_real)
    RI = tf.matmul(theta_real,input_imag)
    IR = tf.matmul(theta_imag,input_real)
    II = tf.matmul(theta_imag,input_imag)
    output_real = RR-II
    output_imag = RI+IR
    outputs = tf.concat([output_real,output_imag],axis=-1)
    return outputs


def E2E_Network(Nt,R,noise_var,trainable_analog,trainable_digital,mode):
    input_data = Input(shape=(Nt,2))

    # add gaussian noise to angular domain channels using tf layers
    # x_real = Lambda(lambda x:x[:,:,0:1])(input_data)
    # x_imag = Lambda(lambda x:x[:,:,1:2])(input_data)
    # x_real = GaussianNoise(stddev=noise_var/2)(x_real)
    # x_imag = GaussianNoise(stddev=noise_var/2)(x_imag)
    # x = Lambda(lambda x: tf.concat(x, axis=-1))([x_real, x_imag])

    x = input_data

    # get the received signal y
    if mode == 'separate':
        h_real,h_imag = B_Layer(Nt)(x)
        h = Lambda(lambda x: tf.concat(x, axis=-1))([h_real, h_imag])
        y = W_Layer(num_inputs=Nt, num_outputs=R, trainable=trainable_analog)(h)
        y = WBB_Layer(num_inputs=R, trainable=trainable_digital)(y)

    if mode == 'joint':
        y = fai_Layer(num_inputs=Nt, num_outputs=R)(x)

    feature_vector = Flatten()(y)
    # add BN before channel estimator to balance the scale of features
    feature_vector = BatchNormalization()(feature_vector)

    # predict the angular domain channel
    feature_vector = Dense(8*Nt,activation='relu')(feature_vector)
    feature_vector = BatchNormalization()(feature_vector)
    #feature_vector = fc_attention_block(feature_vector)
    feature_vector = Dense(4*Nt,activation='relu')(feature_vector)
    feature_vector = BatchNormalization()(feature_vector)
    #feature_vector = fc_attention_block(feature_vector)
    x_hat = Dense(2*Nt)(feature_vector)
    x_hat = Reshape((Nt,2))(x_hat)

    # Conv denoising layers
    # x_hat = Conv1D(kernel_size=5,filters=16,activation='relu',padding='same')(x_hat)
    # x_hat = BatchNormalization()(x_hat)
    # x_hat = Conv1D(kernel_size=1,filters=2)(x_hat)
    
    model = Model(inputs=input_data,outputs=x_hat)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.summary()
    return model



#%% load and construct the dataset and labelset 
Nt = 64
Nr = 4
R = 16

SNR = 20 #dB
SNR_linear = 10**(SNR/10)
noise_var = 1/SNR_linear

data_dir = './data/paper_10_15_%d_4.mat'%Nt
angular_channel_list = io.loadmat(data_dir)['channels']

def DFT_matrix(N):
    F = np.zeros((N,N))+1j*np.zeros((N,N))
    # DFT矩阵每一行对应将角度的sin值进行N等分
    for i in range(-int(N/2),int(N/2)):
        for k in range(N): 
            F[i+int(N/2),k] = np.exp(-1j*2*np.pi*(i+1/2)*k/N)/np.sqrt(N)
    return F
F = DFT_matrix(Nt)
B = np.transpose(np.conjugate(F))

# add noise
noise = np.sqrt(noise_var/2)*(np.random.randn(len(angular_channel_list),Nr,Nt)+1j*(np.random.randn(len(angular_channel_list),Nr,Nt)))
noise_angular = noise.dot(np.transpose(B))
noise_angular = np.transpose(noise_angular,(0,2,1))
angular_channel_list_noisy = angular_channel_list + noise_angular

# slice a part for testing
testing_num = 200
angular_channel_list_testing = angular_channel_list[-testing_num:]
angular_channel_list = angular_channel_list[:-testing_num]
angular_channel_list_noisy_testing = angular_channel_list_noisy[-testing_num:]
angular_channel_list_noisy = angular_channel_list_noisy[:-testing_num]
# LS estimation performance
LS_error = angular_channel_list_noisy_testing - angular_channel_list_testing
LS_error = np.reshape(LS_error,(len(LS_error),Nt*Nr))
angular_channel_list_testing_for_LS = np.reshape(angular_channel_list_testing,(len(angular_channel_list_testing),Nt*Nr))
LS_nmse = np.mean(np.linalg.norm(LS_error,axis=-1)**2/np.linalg.norm(angular_channel_list_testing_for_LS,axis=-1)**2)
print('LS NMSE:',LS_nmse)

# noiseless channel as labelset
labelset = np.expand_dims(angular_channel_list,-1) #data_num,Nt,Nr,1
labelset = np.concatenate([np.real(labelset),np.imag(labelset)],axis=-1)#data_num,Nt,Nr,2
labelset = np.transpose(labelset,(0,2,1,3))#data_num,Nr,Nt,2
labelset = np.reshape(labelset,(len(labelset)*Nr,Nt,2)) # 每连续Nr个是一样的

# noisy channel as dataset
dataset = np.expand_dims(angular_channel_list_noisy,-1)
dataset = np.concatenate([np.real(dataset),np.imag(dataset)],axis=-1)
dataset = np.transpose(dataset,(0,2,1,3))
dataset = np.reshape(dataset,(len(dataset)*Nr,Nt,2))

# shuffle
random_indexs = np.random.choice(range(len(labelset)),len(labelset),replace=False)
labelset = labelset[random_indexs]
dataset = dataset[random_indexs]
print(labelset.shape) #data_num*Nr,Nt,2



#%% construct the network
mode = 'separate'
trainable_analog = True
trainable_digital = True
nn_model = E2E_Network(Nt,R,noise_var,trainable_analog,trainable_digital,mode)

# callbacks
best_model_path = './models/paper_best_%s_10_15_%d_dB.h5'%(mode,SNR)
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,cooldown=2,verbose=1, mode='auto',min_delta=1e-6,min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=1e-6,patience=12)

for i in range(len(nn_model.layers)):
    if nn_model.layers[i].name=='b__layer':
        B_index = i
    if nn_model.layers[i].name=='w__layer':
        W_index = i
    if nn_model.layers[i].name=='wbb__layer':
        WBB_index = i
    if nn_model.layers[i].name=='fai__layer':
        fai_index = i

if mode == 'separate':
    # initialize B layer
    init_weights = np.concatenate([np.real(np.expand_dims(B,-1)),np.imag(np.expand_dims(B,-1))],axis=-1)
    nn_model.layers[B_index].set_weights([init_weights])

    # initialize W layer with ZC matrix
    W = io.loadmat('./data/W_%d_%d.mat'%(Nt,R))['W2']
    init_weights = np.angle(W)
    nn_model.layers[W_index].set_weights([init_weights])

    # initialize WBB layer with I matrix
    identity_matrix = np.eye(R)
    init_weights = np.concatenate([np.real(np.expand_dims(identity_matrix,-1)),np.imag(np.expand_dims(identity_matrix,-1))],axis=-1)
    nn_model.layers[WBB_index].set_weights([init_weights])

if mode=='joint':
    # initialize fai layer with matrix ZC*B
    W = io.loadmat('./data/W_%d_%d.mat'%(Nt,R))['W2']
    init_weights = np.expand_dims(W.dot(B),-1)
    init_weights = np.concatenate([np.real(init_weights),np.imag(init_weights)],axis=-1)
    nn_model.layers[fai_index].set_weights([init_weights])


# fit the autoencoder model
epochs = 1000
batch_size = 128
loss_history = nn_model.fit(dataset,labelset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
             validation_split=0.2,callbacks=[checkpointer,reduce_lr,early_stopping])

# save the loss variation during the training process
io.savemat('./results/loss.mat',{'training loss':loss_history.history['loss'],'validation loss':loss_history.history['val_loss']})

#%% Testing and visualization
# load the best model
nn_model.load_weights(best_model_path)

if mode == 'separate':
    W = np.squeeze(nn_model.layers[W_index].get_weights())
    W = np.cos(W)+1j*np.sin(W)
    WBB = np.squeeze(nn_model.layers[WBB_index].get_weights())
    WBB = WBB[:,:,0]+1j*WBB[:,:,1]
    fai = WBB.dot(W.dot(B))

if mode == 'joint':
    fai = np.squeeze(nn_model.layers[fai_index].get_weights())
    fai = fai[:,:,0] + 1j*fai[:,:,1]
print(fai.shape)

# save data for baseline performance testing
io.savemat('./results/testing_data_%d.mat'%SNR,{'x':angular_channel_list_testing,'x_noisy':angular_channel_list_noisy_testing,'fai':fai})

# visualization
attentions = np.linalg.norm(fai,axis=0)**2
W = io.loadmat('./data/W_%d_%d.mat'%(Nt, R))['W2']
fai_conventional = W.dot(B)
#fai_conventional = W
attentions_conventional = np.linalg.norm(fai_conventional,axis=0)**2
plt.figure()
channel_energy = np.abs(angular_channel_list_testing)
channel_energy = np.transpose(channel_energy,(0,2,1))
channel_energy = np.reshape(channel_energy,(-1,Nt))
channel_energy = np.mean(channel_energy,axis=0)

# plot the normalized energy distribution of channel and the learned fai matrix
plt.plot(channel_energy/np.max(channel_energy), 'r--')
plt.plot(attentions/np.max(attentions),'b-')
plt.plot(attentions_conventional/np.max(attentions_conventional)/2,'g-')
plt.xlim([0,64])
plt.ylim([0,1])
plt.xticks([0,16,32,48,64])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel("Column/Element index")
plt.ylabel("Energy")
plt.legend(['Average angular domain channel','Learned measurement matrix','Conventional measurement matrix'])
plt.grid()
if mode == 'separate':
    save_dir = "./results/paper_result_%s_%s_%s.png"%(mode,trainable_digital,trainable_analog)
if mode == 'joint':
    save_dir = "./results/paper_result_%s.png"%mode
plt.savefig(save_dir)

io.savemat('./results/energies_10_15.mat',{'channel_energy':channel_energy,\
                   'attentions':attentions,'attentions_conventional':attentions_conventional})

# network testing
labelset = np.expand_dims(angular_channel_list_testing,-1)
labelset = np.concatenate([np.real(labelset),np.imag(labelset)],axis=-1)
labelset = np.transpose(labelset,(0,2,1,3))
labelset = np.reshape(labelset,(len(labelset)*Nr,Nt,2))
print(labelset.shape)

noise = np.sqrt(noise_var/2)*(np.random.randn(len(labelset),Nt)+1j*(np.random.randn(len(labelset),Nt)))
noise_angular = np.expand_dims(noise.dot(np.transpose(B)),-1)
dataset = labelset + np.concatenate([np.real(noise_angular),np.imag(noise_angular)],axis=-1)

prediction = nn_model.predict(dataset)
prediction = prediction[:,:,0]+1j*prediction[:,:,1]
prediction = np.reshape(prediction,(len(prediction)//Nr,Nr,Nt))
# save the third sample
io.savemat('nn_prediction.mat',{'nn_prediction':np.transpose(prediction[1])})
prediction = np.transpose(prediction,(0,2,1))
error = prediction - angular_channel_list_testing

# mse
mse = np.linalg.norm(error)**2/np.product(error.shape)
print('MSE:',mse)
# nmse
error = np.reshape(error,(len(error),Nt*Nr))
angular_channel_list_testing = np.reshape(angular_channel_list_testing,(len(angular_channel_list_testing),Nt*Nr))
nmse = np.mean(np.linalg.norm(error,axis=-1)**2/np.linalg.norm(angular_channel_list_testing,axis=-1)**2)
print('NMSE:',nmse)
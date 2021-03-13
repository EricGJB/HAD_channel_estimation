import numpy as np
import matplotlib.pyplot as plt
from scipy import io

Nt = 128
Lp = 10

RF_chain_ratio = 1/4
R = int(RF_chain_ratio*Nt)

print('Consider %d antennas, %d RF chains'%(Nt,R))

channels = io.loadmat('./data/H_list_%d.mat'%Nt)['H_list']

def DFT_matrix(N):
    F = np.zeros((N,N))+1j*np.zeros((N,N))
    # DFT矩阵每一行对应将角度的sin值进行N等分
    for i in range(-int(N/2),int(N/2)):
        for k in range(N): 
            F[i+int(N/2),k] = np.exp(-1j*2*np.pi*(i+1/2)*k/N)/np.sqrt(N)
    return F

F = DFT_matrix(Nt)
B = np.transpose(np.conjugate(F))

# convert channel to angular domain
angular_channels = np.transpose(F.dot(np.transpose(channels)))
print(angular_channels.shape)
data_num = len(angular_channels)

SNR = 10 # dB
SNR_linear = 10**(SNR/10)
noise = np.sqrt(1/SNR_linear)/np.sqrt(2)*(np.random.randn(data_num,Nt)+1j*np.random.randn(data_num,Nt))
angular_noise = np.transpose(F.dot(np.transpose(noise)))
# add noise 
angular_channels_noisy = angular_channels + angular_noise

# show an example of angular domain channel
plt.figure()
plt.plot(np.abs(angular_channels[0]))
plt.plot(np.abs(angular_channels_noisy[0]))

# construct labelset
labelset = np.concatenate([np.real(angular_channels),np.imag(angular_channels)],axis=-1)
training_labelset = labelset[:int(0.8*data_num),:]
testing_labelset = labelset[int(0.8*data_num):,:]

# construct dataset
dataset = np.concatenate([np.real(angular_channels_noisy),np.imag(angular_channels_noisy)],axis=-1)
dataset = np.expand_dims(dataset,-1)
training_dataset = dataset[:int(0.8*data_num),:]
testing_dataset = dataset[int(0.8*data_num):,:]

print(dataset.shape)
print(labelset.shape)


#%% Autoencoder structural network design 
import tensorflow as tf
from tensorflow.keras.layers import Activation,Multiply, GlobalAveragePooling1D, Add, Dense, Conv1D, Flatten, Reshape, Input, BatchNormalization, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
#from keras import backend as K

def attention(x):
    reduction_ratio = 2
    num_filter = int(x.shape[-1])
    num_neurons = num_filter//reduction_ratio
    # squeeze
    x1 = GlobalAveragePooling1D()(x)  
    # attention map prediction
    x2 = Dense(num_neurons,activation='relu')(x1)
    attention_map = Dense(num_filter,activation='sigmoid')(x2)
    # feature recalibration
    x = Multiply()([x,attention_map])
    return x


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
#    print(input_shape)
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]//2),
                                         self.num_outputs]) # 定义一个NtxR的权重张量

  def call(self, input):
    input_real = input[:,:input.shape[-1]//2]
    input_imag = input[:,input.shape[-1]//2:]
    RR = tf.matmul(input_real, tf.cos(self.kernel))
    RI = tf.matmul(input_real, tf.sin(self.kernel))
    IR = tf.matmul(input_imag, tf.cos(self.kernel))
    II = -tf.matmul(input_imag, tf.sin(self.kernel))
    output_real = tf.add(RR,II)
    output_imag = tf.add(RI,IR)
    outputs = tf.concat([output_real,output_imag],axis=-1)
#    print(outputs.shape)
    return outputs


def autoencoder(R,Nt,trainable):    
    ModelInput = Input(shape=(2*Nt,1))    
#    modelinput = Lambda(lambda x:x[:,:,0])(ModelInput)
    
    # multiple matrix B 
    real_times_part = Conv1D(filters=Nt,kernel_size=Nt,strides=Nt,trainable=False,name='Conv1',use_bias=False)(ModelInput)
    imag_times_part = Conv1D(filters=Nt,kernel_size=Nt,strides=Nt,trainable=False,name='Conv2',use_bias=False)(ModelInput)    

    real_Bx_imag = Lambda(lambda x:x[:,1,:])(real_times_part)
    real_Bx_real = Lambda(lambda x:x[:,0,:])(real_times_part)
    imag_Bx_imag = Lambda(lambda x:x[:,1,:])(imag_times_part)
    imag_Bx_real = Lambda(lambda x:x[:,0,:])(imag_times_part)
    
    imag_Bx_imag = Lambda(lambda x:-x)(imag_Bx_imag)

    real = Add()([real_Bx_real,imag_Bx_imag])
    imag = Add()([real_Bx_imag,imag_Bx_real])
    # B x_noisy, of shape (2*Nt,) 
    modelinput = Concatenate()([real,imag]) 
    
    # learn the pure analog matrix with a customized dense layer
    FBWBx = MyDenseLayer(R)(modelinput) # of shape (2*R,)
    
    FBWBx = Reshape((2*R,1))(FBWBx)
    # learn the digital matrix
    real_times_part2 = Conv1D(filters=R,kernel_size=R,strides=R,trainable=trainable,name='Conv3',use_bias=False)(FBWBx)
    imag_times_part2 = Conv1D(filters=R,kernel_size=R,strides=R,trainable=trainable,name='Conv4',use_bias=False)(FBWBx)    

    real_Bx_imag2 = Lambda(lambda x:x[:,1,:])(real_times_part2)
    real_Bx_real2 = Lambda(lambda x:x[:,0,:])(real_times_part2)
    imag_Bx_imag2 = Lambda(lambda x:x[:,1,:])(imag_times_part2)
    imag_Bx_real2 = Lambda(lambda x:x[:,0,:])(imag_times_part2)
    
    imag_Bx_imag2 = Lambda(lambda x:-x)(imag_Bx_imag2)

    real2 = Add()([real_Bx_real2,imag_Bx_imag2])
    imag2 = Add()([real_Bx_imag2,imag_Bx_real2])
    # W_BB W_RF B x_noisy, of shape (2*Nt,) 
    FBWBx2 = Concatenate()([real2,imag2])     
    
    # Channel Estimator
    # Attention aided FNN
    x2 = Dense(16*192,activation='relu')(FBWBx2) 
    x2 = BatchNormalization()(x2)
    # grouping
    x3 = Reshape((16,192))(x2)
    # attention
    x4 = attention(x3)
    xx = Flatten()(x4)
     
    coarse_channels = Dense(2*Nt,activation='linear')(xx)
    prediction = coarse_channels
    
    model = Model(inputs=ModelInput,outputs=prediction)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    model.summary()

    return model

# network parameters
batch_size = 128
epochs = 500
best_model_path = './models/best_%d_%d_%d.h5'%(Nt,R,SNR)

# callbacks
checkpointer = ModelCheckpoint(best_model_path,verbose=1,save_best_only=True,save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,verbose=1, mode='auto',min_delta=1e-5,min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss',min_delta=1e-5,patience=12)

trainable = False # True for joint digital and analog, False for only analog learning
nn_model = autoencoder(R,Nt,trainable)

# load B into the network, and load I into the W_BB as initialization
for layer in nn_model.layers:
    if layer.name == 'Conv1':
        layer.set_weights([np.expand_dims(np.real(np.transpose(B)),axis=1)])
    if layer.name == 'Conv2':
        layer.set_weights([np.expand_dims(np.imag(np.transpose(B)),axis=1)])
    if layer.name in ['Conv3','Conv4']:
        layer.set_weights([np.expand_dims(np.eye(R),axis=1)])
        
loss_history = nn_model.fit(training_dataset,training_labelset,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,\
             validation_split=0.2,callbacks=[checkpointer,early_stopping,reduce_lr])
#io.savemat('./loss_history.mat',{'val_loss':loss_history.history['val_loss']})

# fetch the learned analog matrix
layer_index = 11
my_dense_layer = nn_model.layers[layer_index]
print(my_dense_layer)
learned_weights = my_dense_layer.get_weights()[0] # NtxR
W_RF = np.transpose(np.cos(learned_weights)+1j*np.sin(learned_weights))

# fetch the learned digital matrix
weights = []
for layer in nn_model.layers:
    if layer.name == 'Conv3':
        weights_real = np.squeeze(layer.get_weights())
    if layer.name == 'Conv4':
        weights_imag = np.squeeze(layer.get_weights())
W_BB = np.transpose(weights_real + 1j*weights_imag)

fai = W_BB.dot(W_RF.dot(B))

# save complex data for baseline after training
#received_signal = np.transpose(fai.dot(np.transpose(angular_channels_noisy)))
#io.savemat('./received_signal_%d_%d.mat'%(Nt,R),{'dataset':received_signal,'labelset':angular_channels,'fai':fai})

nn_model.load_weights(best_model_path)
prediction = nn_model.predict(testing_dataset,verbose=1)
prediction_angular_channels = prediction[:,:Nt]+1j*prediction[:,Nt:]
true_angular_channels = angular_channels[int(0.8*(data_num)):]

# NMSE and MSE
#print('Fully digital LS mse:%.4f'%(1/SNR_linear))
error = prediction_angular_channels-true_angular_channels
#nn_nmse = np.mean((np.linalg.norm(error,axis=-1)/np.linalg.norm(true_angular_channels,axis=-1))**2)
nn_mse = np.var(error)
#print('NN nmse:%.4f'%nn_nmse)
print('NN mse:%.4f'%nn_mse)

# Compute the pearson correlation coefficient between two channel vectors 
from scipy.stats.stats import pearsonr  
correlation = 0
for i in range(int(0.2*data_num)):
    correlation = correlation+pearsonr(np.real(prediction_angular_channels[i]),np.real(true_angular_channels[i]))[0]
    correlation = correlation+pearsonr(np.imag(prediction_angular_channels[i]),np.imag(true_angular_channels[i]))[0]
correlation = correlation/int(0.2*data_num)/2
print('Channel correlation:%.4f'%correlation)


#%% plot
signal_energys = np.mean(np.abs(angular_channels),axis=0)
attentions = np.linalg.norm(fai,axis=0)**2
attentions = attentions/np.max(attentions)
plt.figure()
ax=plt.subplot(111)
plt.xlim(0,Nt)
plt.ylim(0,np.max(signal_energys))
plt.xlabel("Column/Element index")
plt.ylabel("Energy")
plt.plot(attentions,'r-')
plt.plot(signal_energys,'m--')

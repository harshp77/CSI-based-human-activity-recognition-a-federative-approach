from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings("ignore")

from npz_loader import *
cfg = CSIModelConfig()
numpy_tuple = cfg.load_csi_data_from_files(('X_bed.npz', 'X_fall.npz', 'X_pickup.npz', 'X_run.npz', 'X_sitdown.npz', 'X_standup.npz', 'X_walk.npz'))
x_bed, y_bed, x_fall, y_fall, x_pickup, y_pickup, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
x_train, y_train, x_valid, y_valid = train_valid_split(
    (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk),
    train_portion=0.9, seed=379)


n_nodes = 10            # Created Nodes for federated learning _ Validation
n_ration = len(x_train)/n_nodes
nodes_dict_train = {}
for i in range(1,n_nodes+1):
    n_ration_f = int(i*n_ration)
    n_ration_b = int((i-1)*n_ration)
    x_train_ = x_train[n_ration_b:n_ration_f,:,:]
    y_train_ = y_train[n_ration_b:n_ration_f,:]
    nodes_dict_train[i] = x_train_,y_train_

print("Nodes Created for training")
print('\n')
n_nodes = 10        # Created Nodes for federated learning _ Validation
n_ration = len(x_valid)/n_nodes
nodes_dict_valid = {}
for i in range(1,n_nodes+1):
    n_ration_f = int(i*n_ration)
    n_ration_b = int((i-1)*n_ration)
    x_valid_ = x_valid[n_ration_b:n_ration_f,:,:]
    y_valid_ = y_valid[n_ration_b:n_ration_f,:]
    nodes_dict_valid[i] = x_valid_,y_valid_
    # print(n_ration_b,n_ration_f)
print("Nodes Created for validation")
print('\n')
# Weight scaling for global averaging
def scale_model_weights(weight, scalar):
    scalar = 1/scalar
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

# weight addition for appending to global model
def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def test(model,x_valid,y_valid):
    arr_pred = np.argmax(model.predict(x_valid,verbose=0),axis=1)
    arr_act = np.argmax(y_valid,axis=1)
    crr = 0
    err = 0
    for i in range(len(x_valid)):
        if arr_pred[i] == arr_act[i]:
            crr += 1
        else :
            err += 1
    print(round(crr/(crr+err),3))
print("functions loaded for learning")
print('\n')

# Parameter Setting
nEpochs = 10  # number of local rounds
comms_round =  30  # number of global round 
k=5  # Number of mandals choosen
earlystop = EarlyStopping(patience=10)

# Global Model Initialization
global_model = modeler(win_len=125)
global_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy', 
    metrics=['accuracy'])
print("Global Model Loaded")
print('\n')
#randomize client data - using keys

client_names= list(nodes_dict_train.keys())

for comm_round in range(comms_round):
            
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    scaled_local_weight_list = list() 
    #loop through each client and create new local model
    for client in client_names:
        local_model = modeler(win_len=125)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy', 
            metrics=['accuracy'])
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        # Data Seperation for min and max input
        (x_train,y_train),(x_valid,y_valid) = nodes_dict_train[client],nodes_dict_valid[client]
        
        history_local = local_model.fit(
                                        x_train,y_train,
                                        batch_size=128, epochs=20,
                                        validation_data=(x_valid, y_valid),
                                        verbose=0,
                                        callbacks=[
                                            tf.keras.callbacks.ModelCheckpoint('best_conv.hdf5',monitor='val_accuracy',
                                                                                save_best_only=True,save_weights_only=False)])
        print('Epoch done for node {} '.format(client))

        #scale the model weights and add to list
        scaled_weights = scale_model_weights(local_model.get_weights(), scalar=k)
        scaled_local_weight_list.append(scaled_weights)

        #clear session to free memory after each communication round
        K.clear_session()
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    print("\n")
    print("Round {} Done for all the nodes".format(comm_round))  

    #update global model 
    global_model.set_weights(average_weights)
    test(global_model,x_valid,y_valid)
    print("\n")
    # test global model and print out metrics after each communications round
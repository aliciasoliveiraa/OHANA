import os
import time
from tqdm import tqdm
from datetime import timedelta
import numpy as np
import tensorflow as tf
from tensorflow import keras
from absl import logging
import matplotlib.pyplot as plt

from nvflare.apis.event_type import EventType
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner

from dataset import datalist_loader, train_batch_data_loader, batch_data_loader
from utils import rot_tra_argumentation, flat_layer_weights_dict, unflat_layer_weights_dict
from model import MC_Net, vgg_layers, make_custom_loss

from fedproxloss import TFFedProxLoss

#CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found. Using CPU")


#MLFLOW
import mlflow

mlflow.enable_system_metrics_logging()

#DEUCALION
mlflow.set_tracking_uri("http://10.2.1.38:5000")

#mlflow.create_experiment(name='MLFlow Moana-FL')
mlflow.set_experiment(experiment_name='MLFlow Moana FedPROX')

class FedLearner(Learner):

    def __init__(
        self,
        lr=1e-4,
        epochs=100,
        batch_size=1,
        image_size=256,
        num_contrast=4,
        num_filter=32,
        num_res_block=9,
        lambda_ssim=1,
        lambda_vgg=1e-2,
        path_logs='logs/',
        path_model='/path/to/moana-fl-fedprox/',
        path_data='/path/to/Data/',
        path_weight='weight/',
        path_saved_models='saved_models',
        reg_type='DatasetFolder',
        save_epoch=10,
        path_training_time='/path/to/moana-fl-fedprox/training_time.txt',
        path_loss_plot='loss_plots/',
        path_accuracy_plot='accuracy_plots/',
        fedproxloss_mu=1e-5
    ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_contrast = num_contrast
        self.num_filter = num_filter
        self.num_res_block = num_res_block
        self.lambda_ssim = lambda_ssim
        self.lambda_vgg = lambda_vgg
        self.path_logs = path_logs
        self.path_model = path_model
        self.path_data = path_data
        self.path_weight = path_weight
        self.path_saved_models = path_saved_models
        self.reg_type = reg_type
        self.save_epoch = save_epoch
        self.path_training_time = path_training_time
        self.path_loss_plot = path_loss_plot
        self.path_accuracy_plot = path_accuracy_plot
        self.fedproxloss_mu = fedproxloss_mu
        self.fedprox_loss_fn = None
    
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self._initialize(fl_ctx)
    
    def _initialize(self, fl_ctx: FLContext):
        logs_path = os.path.join(self.path_model, self.path_logs)
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        logging.get_absl_handler().use_absl_log_file('log', logs_path)
        logging.get_absl_handler().setFormatter(None)

        # Training setup
        self.model = MC_Net(img_size=self.image_size,
                   num_filter=self.num_filter,
                   num_contrast=self.num_contrast,
                   num_res_block=self.num_res_block)
        
        
        self.loss_model = vgg_layers(['block3_conv1'])
        self.final_loss = make_custom_loss(self.lambda_ssim, self.lambda_vgg, self.loss_model)
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr), loss=self.final_loss, metrics=['accuracy'])
        
        for i, layer in enumerate(self.model.layers):
            layer._name = 'layer_' + str(i)

        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.fedprox_loss_fn = TFFedProxLoss(mu=self.fedproxloss_mu)
                
        input_shape = [(None, self.image_size, self.image_size, 1)]
        self.model.build(input_shape=input_shape * self.num_contrast)
        self.model.summary()
        
        self.layer_weights_dict = {layer.name: layer.get_weights() for layer in self.model.layers}

        self.flat_layer_weights_dict = flat_layer_weights_dict(self.layer_weights_dict)
        
        # Data loading
        self.y_train_datalist, self.x_train_datalist = datalist_loader(self.path_data, self.reg_type, 'train')
        self.y_valid_datalist, self.x_valid_datalist = datalist_loader(self.path_data, self.reg_type, 'valid')
        '''
        self.train_size = len(self.y_train_datalist[0])
        #print('self.train_size', self.train_size)
        self.batch_number = int(np.ceil(self.train_size // self.batch_size))
        #print('self.batch_number', self.batch_number)
        self.valid_size = len(self.y_valid_datalist[0])
        #print('self.valid_size', self.valid_size)
        self.batch_number_valid = int(np.ceil(self.valid_size // self.batch_size))
        #print('self.batch_number_valid', self.batch_number_valid)
        '''
        self.iter_interval = 1
        self.min_val_loss = 100000
    
    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        dxo = from_shareable(shareable)
        model_weights = dxo.data
        
        # Log all properties in FLContext
        #self.log_info(fl_ctx, f"FLContext properties: {fl_ctx.props}")
        
        # Obtain the current round from the FLContext
        job_meta = fl_ctx.get_prop("__job_meta__", default={})
        self.min_clients = job_meta.get("min_clients", "Clients property not found")
        
        task_data = fl_ctx.get_prop("__task_data__", default={})
        headers = task_data.get("__headers__", {})
        self.current_round = headers.get("current_round", "Round property not found")
        self.log_info(fl_ctx, f"Starting training for round {self.current_round}")
        
        self.num_rounds = headers.get("num_rounds", "Number of rounds property not found")
        
        self.job_id = fl_ctx.get_job_id()
        
        weights_list = list(model_weights.values())
        
        self.client_name = fl_ctx.get_identity_name()
        client_index = int(self.client_name.split('-')[1]) - 1

        if self.current_round == 0:
            def split_data(datalist, num_clients, client_idx):
                split_size = len(datalist[0]) // num_clients
                start_idx = client_idx * split_size
                end_idx = (client_idx + 1) * split_size if client_idx != num_clients - 1 else len(datalist[0])
                
                split_datalist = [contrast[start_idx:end_idx] for contrast in datalist]
                return split_datalist
            
            self.y_train_datalist = split_data(self.y_train_datalist, self.min_clients, client_index)
            self.x_train_datalist = split_data(self.x_train_datalist, self.min_clients, client_index)
            self.y_valid_datalist = split_data(self.y_valid_datalist, self.min_clients, client_index)
            self.x_valid_datalist = split_data(self.x_valid_datalist, self.min_clients, client_index)
        
        # Log number of images for training
        num_images_train = [len(contrast) for contrast in self.y_train_datalist]
        self.log_info(fl_ctx, f"Client {self.client_name} has {num_images_train} training images.")
        
        self.train_size_client = len(self.y_train_datalist[0])
        self.batch_number_client = int(np.ceil(self.train_size_client // self.batch_size))
        self.valid_size_client = len(self.y_valid_datalist[0])
        self.batch_number_valid_client = int(np.ceil(self.valid_size_client // self.batch_size))
        
        model_global = keras.models.clone_model(self.model)
                
        input_shape = [(None, self.image_size, self.image_size, 1)]
        model_global.build(input_shape=input_shape * self.num_contrast)
        model_global.set_weights(self.model.get_weights())
        
        for layer in model_global.layers:
            layer.trainable = False
                
        print('---------------------JOB ID---------------------', self.job_id)
        print('---------------------CURRENT ROUND---------------------', self.current_round)
        print('---------------------CLIENT NAME---------------------', self.client_name)

        self.local_train(fl_ctx, weights_list, model_global, abort_signal)
        
        # report updated weights in shareable
        new_weights = {layer.name: layer.get_weights() for layer in self.model.layers}
        new_weights_flat = flat_layer_weights_dict(new_weights)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights_flat)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        
        return new_shareable

    def local_train(self, fl_ctx, weights, model_global, abort_signal):
        # Set the model weights
        self.model.set_weights(weights)

        start_training = time.time()
        
        train_loss_history = [[], [], [], []]
        train_accuracy_history = [[], [], [], []]
        
        print('STARTING MODEL TRAINING ...')
        # Model training setup
        for epoch in range(self.epochs):
            self.model.trainable = True
            start_time = time.time()
            train_loss = [0, 0, 0, 0]
            train_accuracy = [0, 0, 0, 0]
            
            y_train_datalist_shuffle, x_train_datalist_shuffle = train_batch_data_loader(self.y_train_datalist + self.x_train_datalist, self.num_contrast)
        
            for batch_index in tqdm(range(self.batch_number_client), ncols=100):
                if abort_signal.triggered:
                    return
                start = batch_index * self.batch_size

                y_train_datalist_batch = []
                x_train_datalist_batch = []
                for i in range(self.num_contrast):
                    y_train_datalist_batch.append(y_train_datalist_shuffle[i][start:start+self.batch_size])
                    x_train_datalist_batch.append(x_train_datalist_shuffle[i][start:start+self.batch_size])

                y_train_batch = batch_data_loader(y_train_datalist_batch, self.num_contrast)
                x_train_batch = batch_data_loader(x_train_datalist_batch, self.num_contrast)  
                y_train_batch, x_train_batch = rot_tra_argumentation(y_train_batch, x_train_batch, self.num_contrast)
                
                batch_size_tmp = x_train_batch[0].shape[0]
                
                tmp_loss = self.model.train_on_batch(x_train_batch, y_train_batch) # the first value is the sum of the train losses
                
                # FedProx loss
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.fedprox_loss_fn(self.model, model_global).numpy()
                    for i in range(1, 5):
                        tmp_loss[i] += fed_prox_loss
                
                if batch_index % self.iter_interval == 0:
                    # Log training information
                    log_message = f'Epoch [{epoch+1:4d}/{self.epochs:4d}] | Iter [{batch_index:4d}/{self.batch_number_client:4d}] ' \
                            f'{time.time() - start_time:.2f}s.. ' \
                            f'train loss for T1: {tmp_loss[1]:.4f}, T1CE: {tmp_loss[2]:.4f}, T2: {tmp_loss[3]:.4f}, FL: {tmp_loss[4]:.4f} & ' \
                            f'Training accuracy for T1: {tmp_loss[5]:.4f}, T1CE: {tmp_loss[6]:.4f}, T2: {tmp_loss[7]:.4f}, FL: {tmp_loss[8]:.4f}'
                    self.log_info(fl_ctx, log_message)

                train_loss = [(x + y * batch_size_tmp) for (x, y) in zip(train_loss, tmp_loss[1:5])]
                train_accuracy = [(x + y * batch_size_tmp) for (x, y) in zip(train_accuracy, tmp_loss[5:9])]

            train_loss = [x / self.train_size_client for x in train_loss]
            train_accuracy = [x / self.train_size_client for x in train_accuracy]

            train_loss_history[0].append(train_loss[0])  # T1
            train_loss_history[1].append(train_loss[1])  # T1CE
            train_loss_history[2].append(train_loss[2])  # T2
            train_loss_history[3].append(train_loss[3])  # FL
            
            train_accuracy_history[0].append(train_accuracy[0])  # T1
            train_accuracy_history[1].append(train_accuracy[1])  # T1CE
            train_accuracy_history[2].append(train_accuracy[2])  # T2
            train_accuracy_history[3].append(train_accuracy[3])  # FL
            
            print(f'Epoch [{epoch+1:4d}/{self.epochs:4d}] {time.time() - start_time:.2f}s.. '
            f'train loss for T1: {train_loss[0]:.4f}, T1CE: {train_loss[1]:.4f}, T2: {train_loss[2]:.4f}, FL: {train_loss[3]:.4f} & '
            f'accuracy for T1: {train_accuracy[0]:.4f}, T1CE: {train_accuracy[1]:.4f}, T2: {train_accuracy[2]:.4f}, FL: {train_accuracy[3]:.4f}') \
            
            log_message = f'Epoch [{epoch+1:4d}/{self.epochs:4d}] {time.time() - start_time:.2f}s.. ' \
                        f'train loss for T1: {train_loss[0]:.4f}, T1CE: {train_loss[1]:.4f}, T2: {train_loss[2]:.4f}, FL: {train_loss[3]:.4f} & ' \
                        f'accuracy for T1: {train_accuracy[0]:.4f}, T1CE: {train_accuracy[1]:.4f}, T2: {train_accuracy[2]:.4f}, FL: {train_accuracy[3]:.4f}'
            self.log_info(fl_ctx, log_message)

            if ((epoch + 1) % self.save_epoch) == 0:                
                valid_loss = [0, 0, 0, 0]
                valid_accuracy = [0, 0, 0, 0]
                y_valid_datalist_shuffle, x_valid_datalist_shuffle = \
                    train_batch_data_loader(self.y_valid_datalist + self.x_valid_datalist, self.num_contrast)
                
                for batch_index in tqdm(range(self.batch_number_valid_client), ncols=100):
                    if abort_signal.triggered:
                        return
                    start_valid = batch_index * self.batch_size

                    y_valid_datalist_batch = []
                    x_valid_datalist_batch = []
                    for i in range(self.num_contrast):
                        y_valid_datalist_batch.append(y_valid_datalist_shuffle[i][start_valid:start_valid+self.batch_size])
                        x_valid_datalist_batch.append(x_valid_datalist_shuffle[i][start_valid:start_valid+self.batch_size])

                    y_valid_batch = batch_data_loader(y_valid_datalist_batch, self.num_contrast)
                    x_valid_batch = batch_data_loader(x_valid_datalist_batch, self.num_contrast)
                    
                    batch_size_tmp_valid = x_valid_batch[0].shape[0]

                    val_loss = self.local_validate(x_valid_batch, y_valid_batch, abort_signal)
                        
                    valid_loss = [(nx + ny * batch_size_tmp_valid) for (nx, ny) in zip(valid_loss, val_loss[1:5])]
                    valid_accuracy = [(x + y * batch_size_tmp) for (x, y) in zip(valid_accuracy, val_loss[5:9])]
                
                valid_loss = [nx / self.valid_size_client for nx in valid_loss]
                valid_accuracy = [x / self.valid_size_client for x in valid_accuracy]

                # Save weights for each 10 epochs
                self.save_weights_dir = os.path.join(self.path_model, self.path_weight)
                if not os.path.exists(self.save_weights_dir):
                    os.makedirs(self.save_weights_dir)
                self.model.save_weights(f'{self.save_weights_dir}weight_e{epoch+1:04d}.h5')
                
                
                # Save model for each 10 epochs
                save_model_dir = os.path.join(self.path_model, self.path_saved_models)
                if not os.path.exists(save_model_dir):
                    os.makedirs(save_model_dir)
                self.model.save(f'{save_model_dir}model_e{epoch + 1:04d}')
                
                if np.mean(valid_loss) < self.min_val_loss:                    
                    path_weight_min_val_loss_dir = os.path.join(self.save_weights_dir, f'weight_min_val_loss_job_{self.job_id}_client_{self.client_name}.h5')
                    path_save_models_dir = os.path.join(save_model_dir, f'model_job_{self.job_id}_client_{self.client_name}')
                    
                    self.model.save_weights(path_weight_min_val_loss_dir, overwrite=True)
                    self.model.save(path_save_models_dir, overwrite=True)

                    self.min_val_loss = np.mean(valid_loss)
                
                with mlflow.start_run(run_name=f'{self.client_name}-epoch_{epoch+1}', log_system_metrics=True) as run:
                    mlflow.log_param("lr", self.lr)
                    mlflow.log_param("epochs", self.epochs)
                    mlflow.log_param("batch_size", self.batch_size)
                    mlflow.log_param("image_size", self.image_size)
                    mlflow.log_param("num_contrast", self.num_contrast)
                    mlflow.log_param("num_filter", self.num_filter)
                    mlflow.log_param("num_res_block", self.num_res_block)
                    mlflow.log_param("lambda_ssim", self.lambda_ssim)
                    mlflow.log_param("lambda_vgg", self.lambda_vgg)
                    mlflow.log_param("path_logs", self.path_logs)
                    mlflow.log_param("path_model", self.path_model)
                    mlflow.log_param("path_data", self.path_data)
                    mlflow.log_param("path_weight", self.path_weight)
                    mlflow.log_param("path_saved_models", self.path_saved_models)
                    mlflow.log_param("reg_type", self.reg_type)
                    mlflow.log_param("save_epoch", self.save_epoch)                      
                    mlflow.log_param("path_training_time", self.path_training_time)
                    mlflow.log_param("path_loss_plot", self.path_loss_plot)
                    mlflow.log_param("path_accuracy_plot", self.path_accuracy_plot)          
                    mlflow.log_param("loss_model", self.loss_model)          
                    mlflow.log_param("final_loss", self.final_loss)          
                    
                    mlflow.log_metric("train_loss T1", train_loss[0])
                    mlflow.log_metric("train_loss T1CE", train_loss[1])
                    mlflow.log_metric("train_loss T2", train_loss[2])
                    mlflow.log_metric("train_loss FL", train_loss[3])

                    mlflow.log_metric("train_accuracy T1", train_accuracy[0])
                    mlflow.log_metric("train_accuracy T1CE", train_accuracy[1])
                    mlflow.log_metric("train_accuracy T2", train_accuracy[2])
                    mlflow.log_metric("train_accuracy FL", train_accuracy[3])
            
                    mlflow.log_metric("valid_loss T1", valid_loss[0])
                    mlflow.log_metric("valid_loss T1CE", valid_loss[1])
                    mlflow.log_metric("valid_loss T2", valid_loss[2])
                    mlflow.log_metric("valid_loss FL", valid_loss[3])
                    
                    mlflow.log_metric("valid_accuracy T1", valid_accuracy[0])
                    mlflow.log_metric("valid_accuracy T1CE", valid_accuracy[1])
                    mlflow.log_metric("valid_accuracy T2", valid_accuracy[2])
                    mlflow.log_metric("valid_accuracy FL", valid_accuracy[3])
                    
                    mlflow.tensorflow.log_model(self.model, artifact_path="model")
                                        
                    mlflow.end_run()
                
                print(f'Weight saved! val loss T1: {valid_loss[0]:.4f}, T1CE: {valid_loss[1]:.4f}, T2: {valid_loss[2]:.4f}, FL: {valid_loss[3]:.4f}')
                print(f'Validation accuracy for T1: {valid_accuracy[0]:.4f}, T1CE: {valid_accuracy[1]:.4f}, T2: {valid_accuracy[2]:.4f}, FL: {valid_accuracy[3]:.4f}')
                
                log_message = f'Weight saved! val loss T1: {valid_loss[0]:.4f}, T1CE: {valid_loss[1]:.4f}, T2: {valid_loss[2]:.4f}, FL: {valid_loss[3]:.4f} & ' \
                            f'Validation accuracy for T1: {valid_accuracy[0]:.4f}, T1CE: {valid_accuracy[1]:.4f}, T2: {valid_accuracy[2]:.4f}, FL: {valid_accuracy[3]:.4f}'
                self.log_info(fl_ctx, log_message)
                del x_valid_batch, y_valid_batch

            
            if epoch+1 == self.epochs:
                path_weight_final_dir = os.path.join(self.save_weights_dir, f'weight_final_job_{self.job_id}_client_{self.client_name}.h5')

                self.model.save_weights(path_weight_final_dir, overwrite=True)

                log_message = f'Weights saved! Training finished.'
                self.log_info(fl_ctx, log_message)
                
        
        end_training = time.time()
        
        training_time = end_training - start_training
        training_time = str(timedelta(seconds=training_time))
        
        with open(self.path_training_time, 'a') as f:
            f.write(f'Training time: {training_time}_job_{self.job_id}_round_{self.current_round}_client_{self.client_name}\n')
        
        with mlflow.start_run(run_name="Training Metrics"):
          
            # Loss plot
            save_loss_dir = os.path.join(self.path_model, self.path_loss_plot)
            if not os.path.exists(save_loss_dir):
                os.makedirs(save_loss_dir)
            
            plt.figure(figsize=(20, 20))

            for i, contrast in enumerate(['T1', 'T1CE', 'T2', 'FL']):
                plt.plot(range(1, self.epochs + 1), train_loss_history[i], label=f'Training Loss ({contrast})')

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_loss_dir}loss_plot_job_{self.job_id}_round_{self.current_round}_client_{self.client_name}.png')
            plt.close()

            mlflow.log_artifact(f'{save_loss_dir}loss_plot_job_{self.job_id}_round_{self.current_round}_client_{self.client_name}.png')

            # Accuracy plot
            save_acc_dir = os.path.join(self.path_model, self.path_accuracy_plot)
            if not os.path.exists(save_acc_dir):
                os.makedirs(save_acc_dir)
            
            plt.figure(figsize=(20, 20))

            for i, contrast in enumerate(['T1', 'T1CE', 'T2', 'FL']):
                plt.plot(range(1, self.epochs + 1), train_accuracy_history[i], label=f'Training Accuracy ({contrast})')

            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy over Epochs')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{save_acc_dir}accuracy_plot_job_{self.job_id}_round_{self.current_round}_client_{self.client_name}.png')
            plt.close()
                
            mlflow.log_artifact(f'{save_acc_dir}accuracy_plot_job_{self.job_id}_round_{self.current_round}_client_{self.client_name}.png')
                
            mlflow.end_run()


    def local_validate(self, x_valid_batch, y_valid_batch, abort_signal):
        self.model.trainable = False
        if abort_signal.triggered:
            return 0
        
        val_loss = self.model.evaluate(x_valid_batch, y_valid_batch, verbose=0)

        return val_loss
    

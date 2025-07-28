import os
import tensorflow as tf
from tensorflow import keras
import shutil

from contextlib import redirect_stdout, redirect_stderr

from model import MC_Net, vgg_layers, make_custom_loss
from utils import flat_layer_weights_dict, unflat_layer_weights_dict
from dataset import datalist_loader, batch_data_loader
from utils import test_ssim, test_nmi, test_nrmse, save_image

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor

#CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found. Using CPU")

#MEMORY ALLOCATOR
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class TFModelPersistor(ModelPersistor):
    def __init__(self, save_name="tf_model.h5"):
        super().__init__()
        self.save_name = save_name
        self.model = None
        self.batch_size=1
        self.lr=1e-4
        self.image_size=256
        self.num_contrast=4
        self.lambda_ssim=1
        self.lambda_vgg=1e-2
        self.num_filter=32
        self.num_res_block=9
        self.path_model='/path/to/moana-fl-fedprox/'
        self.path_weight='weight/'
        self.load_weight_name=None
        self.path_data="/path/to/Data"
        self.path_save_test=None
        self.reg_type="DatasetFolder"

    def _initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_root = workspace.get_app_dir(fl_ctx.get_job_id())
        self._model_save_path = os.path.join(app_root, self.save_name)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        if os.path.exists(self._model_save_path):
            self.logger.info("Loading server model and weights")
            self.model.load_weights(self._model_save_path)
            print("Weights loaded successfully.")
            
        else:
            print("Model file does not exist. Initializing server model...")
            self.logger.info("Initializing server model")
            self.model = MC_Net(img_size=self.image_size,
                   num_filter=self.num_filter,
                   num_contrast=self.num_contrast,
                   num_res_block=self.num_res_block)
            
            self.loss_model = vgg_layers(['block3_conv1'])
            self.final_loss = make_custom_loss(self.lambda_ssim, self.lambda_vgg, self.loss_model)
            self.model.compile(optimizer=keras.optimizers.Adam(self.lr), loss=self.final_loss, metrics=['accuracy'])

            for i, layer in enumerate(self.model.layers):
                layer._name = 'layer_' + str(i)
            
            input_shape = [(None, self.image_size, self.image_size, 1)]
            self.model.build(input_shape=input_shape * self.num_contrast)
                        
            self.layer_weights_dict = {layer.name: layer.get_weights() for layer in self.model.layers}
            self.flat_layer_weights_dict = flat_layer_weights_dict(self.layer_weights_dict)

            model_learnable = make_model_learnable(self.flat_layer_weights_dict, dict())
            
            self.log_info(fl_ctx, f"FLContext properties: {fl_ctx.props}")
            
            job_meta = fl_ctx.get_prop("__job_meta__", default={})
            self.min_clients = job_meta.get("min_clients", "Clients property not found")
            
            self.num_rounds = fl_ctx.get_prop("num_rounds", "Rounds property not found")
            
        return model_learnable

    def _evaluate_model_client(self, fl_ctx: FLContext):
        result_file_path = f'/path/to/moana-fl-fedprox/result_test_job_{fl_ctx.get_job_id()}_clients.txt'
        
        job_meta = fl_ctx.get_prop("__job_meta__", default={})
        self.min_clients = job_meta.get("min_clients", "Clients property not found")
        
        clients = [f'site-{i+1}' for i in range(self.min_clients)]
        
        for client in clients:
            with open(result_file_path, 'a') as result_file, redirect_stdout(result_file), redirect_stderr(result_file):
                print('-------------------START-------------------')
                print(f'Client: {client}')

                self.model = MC_Net(img_size=self.image_size,
                                    num_filter=self.num_filter,
                                    num_contrast=self.num_contrast,
                                    num_res_block=self.num_res_block)
                
                input_shape = [(None, self.image_size, self.image_size, 1)]
                self.model.build(input_shape=input_shape * self.num_contrast)

                self.load_weight_name = f'weight_final_job_{fl_ctx.get_job_id()}_client_{client}.h5'
                print('load_weight_name', self.load_weight_name)
                
                self.model.load_weights(self.save_weights_dir + self.load_weight_name)
                print('Model building completed!')
                
                y_test_datalist, x_test_datalist = datalist_loader(self.path_data, self.reg_type, 'test')
                x_test = batch_data_loader(x_test_datalist, self.num_contrast)
                y_test = batch_data_loader(y_test_datalist, self.num_contrast)
                print('Data loading completed!')
                
                p_test = self.model.predict(x_test, batch_size=self.batch_size)
                print('Prediction completed!')
                
                x_ssim_T1, p_ssim_T1 = test_ssim(x_test[0], y_test[0], p_test[0])
                x_ssim_T1CE, p_ssim_T1CE = test_ssim(x_test[1], y_test[1], p_test[1])
                x_ssim_T2, p_ssim_T2 = test_ssim(x_test[2], y_test[2], p_test[2])
                x_ssim_FL, p_ssim_FL = test_ssim(x_test[3], y_test[3], p_test[3])

                print(f'c    | x_ssim   | p_ssim')
                print(f'T1   | {x_ssim_T1:.4f}   | {p_ssim_T1:.4f}')
                print(f'T1CE | {x_ssim_T1CE:.4f}   | {p_ssim_T1CE:.4f}')
                print(f'T2   | {x_ssim_T2:.4f}   | {p_ssim_T2:.4f}')
                print(f'FL   | {x_ssim_FL:.4f}   | {p_ssim_FL:.4f}')
                print('')
                
                x_nmi_T1, p_nmi_T1 = test_nmi(x_test[0], y_test[0], p_test[0])
                x_nmi_T1CE, p_nmi_T1CE = test_nmi(x_test[1], y_test[1], p_test[1])
                x_nmi_T2, p_nmi_T2 = test_nmi(x_test[2], y_test[2], p_test[2])
                x_nmi_FL, p_nmi_FL = test_nmi(x_test[3], y_test[3], p_test[3])

                print(f'c    | x_nmi    | p_nmi ')
                print(f'T1   | {x_nmi_T1:.4f}   | {p_nmi_T1:.4f}')
                print(f'T1CE | {x_nmi_T1CE:.4f}   | {p_nmi_T1CE:.4f}')
                print(f'T2   | {x_nmi_T2:.4f}   | {p_nmi_T2:.4f}')
                print(f'FL   | {x_nmi_FL:.4f}   | {p_nmi_FL:.4f}')
                print('')

                x_nrmse_T1, p_nrmse_T1 = test_nrmse(x_test[0], y_test[0], p_test[0])
                x_nrmse_T1CE, p_nrmse_T1CE = test_nrmse(x_test[1], y_test[1], p_test[1])
                x_nrmse_T2, p_nrmse_T2 = test_nrmse(x_test[2], y_test[2], p_test[2])
                x_nrmse_FL, p_nrmse_FL = test_nrmse(x_test[3], y_test[3], p_test[3])

                print(f'c    | x_nrmse  | p_nrmse')
                print(f'T1   | {x_nrmse_T1:.4f}   | {p_nrmse_T1:.4f}')
                print(f'T1CE | {x_nrmse_T1CE:.4f}   | {p_nrmse_T1CE:.4f}')
                print(f'T2   | {x_nrmse_T2:.4f}   | {p_nrmse_T2:.4f}')
                print(f'FL   | {x_nrmse_FL:.4f}   | {p_nrmse_FL:.4f}')
                print('')
                
                print(f"{p_ssim_T1:.4f},{p_ssim_T1CE:.4f},{p_ssim_T2:.4f},{p_ssim_FL:.4f},{p_nmi_T1:.4f},{p_nmi_T1CE:.4f},{p_nmi_T2:.4f},{p_nmi_FL:.4f},{p_nrmse_T1:.4f},{p_nrmse_T1CE:.4f},{p_nrmse_T2:.4f},{p_nrmse_FL:.4f}")

                print('-------------------END-------------------')
                
    def _test_model(self, fl_ctx: FLContext):
        log_file_path = f'/path/to/moana-fl-fedprox/result_test_job_{fl_ctx.get_job_id()}.txt'
        with open(log_file_path, 'a') as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
            print('-------------------START-------------------')
            self.model = MC_Net(img_size=self.image_size,
                                num_filter=self.num_filter,
                                num_contrast=self.num_contrast,
                                num_res_block=self.num_res_block)

            input_shape = [(None, self.image_size, self.image_size, 1)]
            self.model.build(input_shape=input_shape * self.num_contrast)
            
            self.load_weight_name=f'job_{fl_ctx.get_job_id()}.h5'
            print('load_weight_name', self.load_weight_name)
            self.path_save_test= f'/path/to/moana-fl-fedprox/test_job_{fl_ctx.get_job_id()}'
            print('path_save_test', self.path_save_test)
            
            self.model.load_weights(self.save_weights_dir + self.load_weight_name)
            print('Model building completed!')

            y_test_datalist, x_test_datalist = datalist_loader(self.path_data, self.reg_type, 'test')
            x_test = batch_data_loader(x_test_datalist, self.num_contrast)
            y_test = batch_data_loader(y_test_datalist, self.num_contrast)
            print('Data loading completed!')
            
            p_test = self.model.predict(x_test, batch_size=self.batch_size)
            print('Prediction completed!')
            
            x_ssim_T1, p_ssim_T1 = test_ssim(x_test[0], y_test[0], p_test[0])
            x_ssim_T1CE, p_ssim_T1CE = test_ssim(x_test[1], y_test[1], p_test[1])
            x_ssim_T2, p_ssim_T2 = test_ssim(x_test[2], y_test[2], p_test[2])
            x_ssim_FL, p_ssim_FL = test_ssim(x_test[3], y_test[3], p_test[3])

            print(f'c    | x_ssim   | p_ssim')
            print(f'T1   | {x_ssim_T1:.4f}   | {p_ssim_T1:.4f}')
            print(f'T1CE | {x_ssim_T1CE:.4f}   | {p_ssim_T1CE:.4f}')
            print(f'T2   | {x_ssim_T2:.4f}   | {p_ssim_T2:.4f}')
            print(f'FL   | {x_ssim_FL:.4f}   | {p_ssim_FL:.4f}')
            print('')
            x_nmi_T1, p_nmi_T1 = test_nmi(x_test[0], y_test[0], p_test[0])
            x_nmi_T1CE, p_nmi_T1CE = test_nmi(x_test[1], y_test[1], p_test[1])
            x_nmi_T2, p_nmi_T2 = test_nmi(x_test[2], y_test[2], p_test[2])
            x_nmi_FL, p_nmi_FL = test_nmi(x_test[3], y_test[3], p_test[3])

            print(f'c    | x_nmi    | p_nmi ')
            print(f'T1   | {x_nmi_T1:.4f}   | {p_nmi_T1:.4f}')
            print(f'T1CE | {x_nmi_T1CE:.4f}   | {p_nmi_T1CE:.4f}')
            print(f'T2   | {x_nmi_T2:.4f}   | {p_nmi_T2:.4f}')
            print(f'FL   | {x_nmi_FL:.4f}   | {p_nmi_FL:.4f}')
            print('')

            x_nrmse_T1, p_nrmse_T1 = test_nrmse(x_test[0], y_test[0], p_test[0])
            x_nrmse_T1CE, p_nrmse_T1CE = test_nrmse(x_test[1], y_test[1], p_test[1])
            x_nrmse_T2, p_nrmse_T2 = test_nrmse(x_test[2], y_test[2], p_test[2])
            x_nrmse_FL, p_nrmse_FL = test_nrmse(x_test[3], y_test[3], p_test[3])

            print(f'c    | x_nrmse  | p_nrmse')
            print(f'T1   | {x_nrmse_T1:.4f}   | {p_nrmse_T1:.4f}')
            print(f'T1CE | {x_nrmse_T1CE:.4f}   | {p_nrmse_T1CE:.4f}')
            print(f'T2   | {x_nrmse_T2:.4f}   | {p_nrmse_T2:.4f}')
            print(f'FL   | {x_nrmse_FL:.4f}   | {p_nrmse_FL:.4f}')
            print('')
            
            print(f"{p_ssim_T1:.4f},{p_ssim_T1CE:.4f},{p_ssim_T2:.4f},{p_ssim_FL:.4f},{p_nmi_T1:.4f},{p_nmi_T1CE:.4f},{p_nmi_T2:.4f},{p_nmi_FL:.4f},{p_nrmse_T1:.4f},{p_nrmse_T1CE:.4f},{p_nrmse_T2:.4f},{p_nrmse_FL:.4f}")

            os.makedirs(self.path_save_test, exist_ok=True)
            for i in range(p_test[0].shape[0]):
                save_image(f'{self.path_save_test}/T1_pred_{i+1:04d}.png', p_test[0][i])
                save_image(f'{self.path_save_test}/T1CE_pred_{i+1:04d}.png', p_test[1][i])
                save_image(f'{self.path_save_test}/T2_pred_{i+1:04d}.png', p_test[2][i])
                save_image(f'{self.path_save_test}/FL_pred_{i+1:04d}.png', p_test[3][i])
            print('Image saving completed!')
            print('-------------------END-------------------')

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    
    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        result = unflat_layer_weights_dict(model_learnable[ModelLearnableKey.WEIGHTS])
        
        for i, layer in enumerate(self.model.layers):
            layer._name = 'layer_' + str(i)

        for k in result:
            layer = self.model.get_layer(name=k)
            layer.set_weights(result[k])
        
        self.model.save_weights(self._model_save_path)
        self.save_weights_dir = os.path.join(self.path_model, self.path_weight)
        if not os.path.exists(self.save_weights_dir):
            os.makedirs(self.save_weights_dir)
        shutil.copy(self._model_save_path, f'{self.save_weights_dir}job_{fl_ctx.get_job_id()}.h5')
        log_message = f'Model saved!'
        self.log_info(fl_ctx, log_message)
        
        self._evaluate_model_client(fl_ctx)
        self.log_info(fl_ctx, f"Model Client tested!")
                
        self._test_model(fl_ctx)
        log_message = f'Model tested!'
        self.log_info(fl_ctx, log_message)
        
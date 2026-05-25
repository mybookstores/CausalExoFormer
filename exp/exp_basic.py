import os
import torch
import importlib
import pkgutil  

# Just put your model files under models/ folder
# e.g., models/Transformer.py, models/LSTM.py, etc.
# All models will be automatically detected and can be used by specifying their names.

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        
        # -------------------------------------------------------
        #  Automatically generate model map
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # Use smart dictionary
        self.model_dict = LazyModelDict(model_map)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """
        Automatically scan all .py files in the models folder
        """
        model_map = {}
        models_dir = 'models'

        # Iterate through all files in 'models' directory
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # Ignore __init__.py and non-.py files
                if filename.endswith('.py') and filename != '__init__.py':
                    # Remove .py extension to get module name
                    module_name = filename[:-3]
                    
                    # Build full import path
                    full_path = f"{models_dir}.{module_name}"
                    
                    # loading dict: {'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path
        
        return model_map

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # IMPORTANT:
            # This project sets CUDA_VISIBLE_DEVICES *inside* the process to select GPUs.
            # After doing so, CUDA device indices become *local* (0..N-1), not global.
            # Therefore, we must map the selected GPU to local index 0.
            if not self.args.use_multi_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                self.args.gpu = 0
                device = torch.device('cuda:0')
                self.args.device = device
                print('Use GPU: cuda:0')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                # After masking, device_ids should be local indices [0..k-1]
                device_ids = [d for d in self.args.devices.replace(' ', '').split(',') if d]
                self.args.device_ids = list(range(len(device_ids)))
                self.args.gpu = 0
                device = torch.device('cuda:0')
                self.args.device = device
                print('Use GPU: cuda:0 (multi)')
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            self.args.device = device
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            self.args.device = device
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class LazyModelDict(dict):
    """
    Smart Lazy-Loading Dictionary
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")
            
        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...") 
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # Try to find the model class
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class


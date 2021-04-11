import os
import pathlib
import shutil
from abc import ABC, abstractmethod



class LoggerWriter(ABC):
    def __init__(self, project_name, enable_log, cfg):
        self.support_type = ['scalar', 'matrix']
        self.project_name = project_name
        self.enable_log = enable_log
        self.cfg = cfg

    @abstractmethod
    def get_run_name(self, dict):
        pass

    @abstractmethod
    def log_params(self, dict):
        pass
    
    @abstractmethod
    def log_metric(self, name, value, step):
        pass

    @abstractmethod
    def log_confusion_matrix(self, matrix, step):
        pass

    @abstractmethod
    def log_asset(self, file):
        pass
     

class CometWriter(LoggerWriter):
    def __init__(self, project_name, enable_log, cfg):
        from comet_ml import Experiment
        super().__init__(project_name, enable_log, cfg)
        self.exp = Experiment(api_key=cfg.comet_api_key, project_name=project_name, disabled= not enable_log)
    
    def log_params(self, args):
        self.exp.log_parameters(vars(args))
    
    def log_metric(self, name, value, step):
        self.exp.log_metric(name, value, step)
    
    def log_confusion_matrix(self, matrix, step):
        self.exp.log_confusion_matrix(matrix=matrix, step=step, file_name="confusion_matrix_{}".format(step))
    

class WandBWriter(LoggerWriter):
    def __init__(self, project_name, enable_log, cfg):
        super().__init__(project_name, enable_log, cfg)
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
        #dryrun 和 run的区别是什么？
        os.environ["WANDB_MODE"] = "run" if enable_log else "dryrun"
        log_path = os.environ['SVAE_LOG']
        print(project_name, log_path)
        import wandb
        wandb.init(project=project_name, dir=log_path)
        self.wandb = wandb
        self.epoch = -1
        self.step = -1

    def _log_epoch(self, epoch):
        if epoch != self.epoch:
            self.step = self.step + 1
            self.wandb.log({'Epoch': epoch}, step=self.step)
            self.epoch = epoch
    
    def log_params(self, args):
        self.wandb.config.update(vars(args), allow_val_change=True)
    
    def log_metric(self, name, value, step):
        self._log_epoch(step)
        self.wandb.log({name:value}, step=self.step)
    
    def log_confusion_matrix(self, matrix, step):
        labels = []
        preds = []
        for label, row in enumerate(matrix):
            for pred, count in enumerate(row):
                labels += count * [label,]
                preds += count * [pred,]
        self.wandb.sklearn.plot_confusion_matrix(labels, preds)
    
    def log_asset(self, file):
        #
        sscr = pathlib.Path(file)
        print(sscr)
        dscr = pathlib.Path(self.wandb.run.dir).joinpath(sscr.name)
        shutil.copyfile(sscr, dscr)
        dscr = str(dscr)
        self.wandb.save(dscr)
    
    def get_run_name(self):
        return self.wandb.run.name


    
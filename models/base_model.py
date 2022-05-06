import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from fvcore import parameter_count, parameter_count_table

from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (dict like class)-- stores all the experiment flags

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See aoda_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_id = opt.gpu_id
        self.isTrain = opt.isTrain
        self.device = (
            torch.device("cuda:{}".format(self.gpu_id))
            if self.gpu_id
            else torch.device("cpu")
        )  # get device name: CPU or GPU
        self.save_dir = os.path.join(
            opt.checkpoints_dir, opt.name
        )  # save all the checkpoints to save_dir

        #? with [scale_width], input imgs might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != "scale_width":  
            torch.backends.cudnn.benchmark = True

        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.Tensor = torch.cuda.FloatTensor if self.gpu_id else torch.Tensor

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [
                networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers
            ]
        if not self.isTrain or opt.continue_train:
            load_suffix = "iter_%d" % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self): #TODO ?
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """为所有网络更新学习率; 在每个epoch的末尾调用"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def get_current_visuals(self):
        """返回需要可视化的图像"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """返回定义的各种loss"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """保存模型到 ./checkpoints/name/models/

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(
                    self.save_dir, "models", save_filename)
                net = getattr(self, "net" + name)

                if self.gpu_id == 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_id)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def save_training_state(self, epoch, iter_step):
        """保存训练状态, 用于恢复训练状态"""
        state = {"epoch": epoch, "iter": iter_step,
                 "schedulers": [], "optimizers": []}
        for s in self.schedulers:
            state["schedulers"].append(s.state_dict())
        for o in self.optimizers:
            state["optimizers"].append(o.state_dict())
        save_filename = "latest.state"
        save_path = os.path.join(self.save_dir, "states", save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """恢复训练状态，包括schedulers, optimizers"""
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(self.optimizers), "Wrong lengths of optimizers"
        assert len(resume_schedulers) == len(self.schedulers), "Wrong lengths of schedulers"
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    #TODO 能直接删吗
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    #TODO 改
    def load_pretrain_from_path(self, path, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = "%s_net_%s.pth" % (epoch, name)
                load_path = os.path.join(path, "models", load_filename)
                net = getattr(self, "net" + name)
                print("loading the model from %s" % load_path)
                state_dict = torch.load(
                    load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(
                    state_dict.keys()
                ):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(
                        state_dict, net, key.split(".")
                    )
                net.load_state_dict(state_dict)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:#TODO ?
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        self.load_pretrain_from_path(self.save_dir, epoch)

    def print_networks(self, verbose):
        """打印网络的参数总量

        Parameters:
            verbose (bool) -- if verbose: print the network parmeters per layer
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = parameter_count(net)['']
                if verbose:
                    print(parameter_count_table(net))
                print(
                    "[Network %s] Total number of parameters : %.3f M"
                    % (name, num_params / 1e6)
                )
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """冻结指定网络
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

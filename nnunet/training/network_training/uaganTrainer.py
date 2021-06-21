from _warnings import warn
import matplotlib
from nnunet.training.network_training.network_trainer import NetworkTrainer
from nnunet.network_architecture.neural_network import SegmentationNetwork
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.nd_softmax import softmax_helper
import torch
import numpy as np
from time import time
import torch.backends.cudnn as cudnn
import itertools
from nnunet.utilities.tensor_utilities import sum_tensor
from torch.optim import lr_scheduler
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.network_architecture.uagan import UAGAN, Discriminator
# from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from torch import nn
import torch.nn.functional as F
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, \
    default_2D_augmentation_params, get_default_augmentation, get_patch_size
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.evaluation.evaluator import aggregate_scores
from multiprocessing import Pool
from nnunet.evaluation.metrics import ConfusionMatrix

matplotlib.use("agg")
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from nnunet.paths import caseid2modal, modal2index, m_dim


class uaganTrainer(NetworkTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super(uaganTrainer, self).__init__(deterministic, fp16)
        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16)
        # set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder
        self.fold = fold

        self.plans = None
        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None

        self.folder_with_preprocessed_data = None

        # set in self.initialize()

        self.dl_tr = self.dl_val = None
        self.num_input_channels = self.num_classes = self.net_pool_per_axis = self.patch_size = self.batch_size = \
            self.threeD = self.base_num_features = self.intensity_properties = self.normalization_schemes = \
            self.net_num_pool_op_kernel_sizes = self.net_conv_kernel_sizes = None  # loaded automatically from plans_file
        self.basic_generator_patch_size = self.data_aug_params = self.transpose_forward = self.transpose_backward = None

        self.batch_dice = batch_dice
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {})

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.classes = self.do_dummy_2D_aug = self.use_mask_for_norm = self.only_keep_largest_connected_component = \
            self.min_region_size_per_class = self.min_size_per_class = None

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.update_fold(fold)
        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30

        self.initial_lr = 3e-4
        self.min_lr = 3e-7
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        # New Add.
        self.iteration = 0
        self.D = None
        self.d_optimizer = None

        self.lambda_cls = 10.
        self.lambda_gp = 10.
        self.lambda_rec = 100.
        self.lambda_seg = 100.
        self.lambda_shape = 0.
        self.print_freq = 50
        self.reduction = 6
        self.writer = SummaryWriter()

        # over write
        self.max_num_epochs = 300
        self.num_batches_per_epoch = 150
        self.num_val_batches_per_epoch = 10  # 30
        need_to_print = dict(lr=self.initial_lr, min_lr=self.min_lr, lambda_cls=self.lambda_cls,
                             lambda_gp=self.lambda_gp, lambda_rec=self.lambda_rec, lambda_seg=self.lambda_seg,
                             lambda_shape=self.lambda_shape, max_epoch=self.max_num_epochs,
                             num_batches_per_epoch=self.num_batches_per_epoch,
                             num_val_batches_per_epoch=self.num_val_batches_per_epoch,
                             reduction=self.reduction)
        self.print_to_log_file(need_to_print)

    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def setup_DA_params(self):
        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

    def initialize(self, training=True, force_load_plans=False):
        """
        For prediction of test cases just set training=False, this will prevent loading of training data and
        training batchgenerator initialization
        :param training:
        :return:
        """

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or (self.plans is None):
            self.load_plans_file()

        self.process_plans(self.plans)

        self.setup_DA_params()  # 设置数据增强的参数
        self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                  "_stage%d" % self.stage)
        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                self.print_to_log_file("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                self.print_to_log_file("done")
            else:
                self.print_to_log_file(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")
            self.tr_gen, self.val_gen = get_default_augmentation(self.dl_tr, self.dl_val,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)
            self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                   also_print_to_console=False)
            # self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
            #                        also_print_to_console=False)
        else:
            pass
        self.initialize_network_optimizer_and_scheduler()
        # assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    def initialize_network_optimizer_and_scheduler(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        base_num_features = 12
        self.print_to_log_file(f'base_num_features: {base_num_features}')
        self.network = UAGAN(self.num_input_channels + m_dim, base_num_features, self.num_classes, net_numpool,
                             2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                             net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                             self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True,
                             reduction=self.reduction,
                             atten_bitmap=[0, 0, 0, 0, 1])
        self.print_to_log_file(self.network.cross_atten)

        self.D = Discriminator(self.patch_size, weightInitializer=InitWeights_He(1e-2), repeat_num=5)

        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                            amsgrad=True)
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_num_epochs,
                                                           eta_min=self.min_lr)
        self.network.cuda()
        self.D.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def run_training(self):
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

        super(uaganTrainer, self).run_training()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """The interface of the train and val mode.
        """
        if do_backprop:
            return self.run_iteration_train(data_generator, do_backprop, run_online_evaluation)
        else:
            return self.run_iteration_val(data_generator, do_backprop, run_online_evaluation)

    def lambda_shape_step(self):
        if self.lambda_shape < self.lambda_seg and self.iteration % self.num_batches_per_epoch == 0:
            self.lambda_shape += 1
        if self.iteration % self.num_batches_per_epoch == 0:
            self.print_to_log_file(self.lambda_shape)

    def run_iteration_train(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """ Re-implement the run iteration to UAGAN version.
        """
        self.iteration += 1
        loss = OrderedDict()
        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #
        data_dict = next(data_generator)
        labels_org = data_dict['mindex']
        labels_trg = np.array([np.random.randint(0, m_dim) for _ in labels_org])

        x_real = data_dict['data']
        vec_org = self.label2onenot(labels_org, m_dim)
        vec_trg = self.label2onenot(labels_trg, m_dim)
        mask = data_dict['target']

        labels_org = torch.from_numpy(labels_org).long()
        labels_trg = torch.from_numpy(labels_trg).long()
        vec_org = torch.from_numpy(vec_org).float()
        vec_trg = torch.from_numpy(vec_trg).float()
        if not isinstance(x_real, torch.Tensor):
            x_real = torch.from_numpy(x_real).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        labels_org = labels_org.cuda()
        labels_trg = labels_trg.cuda()
        vec_org = vec_org.cuda()
        vec_trg = vec_trg.cuda()
        x_real = x_real.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #
        # real images.
        out_org, out_cls = self.D(x_real)
        d_loss_real = - torch.mean(out_org)
        d_loss_cls = F.cross_entropy(out_cls, labels_org)

        # fake images.
        _, x_fake = self.network(x_real, vec_org, vec_trg, output_tsl=True)
        out_src, _ = self.D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1, 1).cuda()
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(out_src, x_hat)

        # backward.
        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
        if do_backprop:
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
        loss['D/fake'] = d_loss_fake.item()
        loss['D/real'] = d_loss_real.item()
        loss['D/cls'] = d_loss_cls.item()
        loss['D/gp'] = d_loss_gp.item()

        del x_fake, x_hat
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        # org-trg domain.
        pred_ot, x_fake = self.network(x_real, vec_org, vec_trg, output_tsl=True)
        g_loss_seg = self.loss(pred_ot, mask)

        out_src, out_cls = self.D(x_fake)
        g_loss_fake = - torch.mean(out_src)
        g_loss_cls = F.cross_entropy(out_cls, labels_trg)

        if run_online_evaluation:
            self.run_online_evaluation(pred_ot, mask)

        # trg-org domain.
        pred_to, x_rec = self.network(x_fake, vec_trg, vec_org, output_tsl=True)
        g_loss_rec = torch.mean(torch.abs(x_real - x_rec))
        g_loss_shape = self.loss(pred_to, mask)

        # backward.
        g_loss = g_loss_fake + self.lambda_cls * g_loss_cls + self.lambda_seg * g_loss_seg + \
                self.lambda_rec * g_loss_rec + self.lambda_shape * g_loss_shape
        loss['G/lambda_shape'] = self.lambda_shape
        self.lambda_shape_step()

        if do_backprop:
            self.optimizer.zero_grad()
            g_loss.backward()
            self.optimizer.step()

        loss['G/fake'] = g_loss_fake.item()
        loss['G/cls'] = g_loss_cls.item()
        loss['G/seg'] = g_loss_seg.item()
        loss['G/rec'] = g_loss_rec.item()
        loss['G/shape'] = g_loss_shape.item()

        del pred_ot, pred_to, x_fake, x_rec, x_real, mask
        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        if self.iteration % self.print_freq == 0:
            # print(loss)
            for k, v in loss.items():
                self.writer.add_scalar(k, v, self.iteration)
        return g_loss_seg.detach().cpu().numpy()

    def run_iteration_val(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """ Re-implement the run iteration to UAGAN version.
        """
        data_dict = next(data_generator)

        x_real = data_dict['data']
        mask = data_dict['target']
        if not isinstance(x_real, torch.Tensor):
            x_real = torch.from_numpy(x_real).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        x_real = x_real.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        pred_ot = self.network(x_real)
        g_loss_seg = self.loss(pred_ot, mask)
        if run_online_evaluation:
            self.run_online_evaluation(pred_ot, mask)
        del x_real, mask
        return g_loss_seg.detach().cpu().numpy()

    @staticmethod
    def label2onenot(labels, dim):
        batch_size = labels.shape[0]
        out = np.zeros((batch_size, dim))
        out[np.arange(batch_size), labels] = 1
        return out

    @staticmethod
    def gradient_penalty(y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        """ We directly change the args in here rather than the plans.pkl. It is not an elegant way.
        """
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
        self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
        self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                               "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                                   "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

    def load_dataset(self):
        self.dataset = load_dataset(self.folder_with_preprocessed_data)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 transpose=None,  # self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  transpose=None,  # self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnunet.preprocessing.preprocessing import GenericPreprocessor, PreprocessorFor2D
        if self.threeD:
            preprocessor = GenericPreprocessor(self.normalization_schemes, self.use_mask_for_norm,
                                               self.transpose_forward, self.intensity_properties)
        else:
            preprocessor = PreprocessorFor2D(self.normalization_schemes, self.use_mask_for_norm,
                                             self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties

    def preprocess_predict_nifti(self, input_files, output_file=None, softmax_ouput_file=None):
        """
        Use this to predict new data
        :param input_files:
        :param output_file:
        :param softmax_ouput_file:
        :return:
        """
        print("preprocessing...")
        d, s, properties = self.preprocess_patient(input_files)
        print("predicting...")
        pred = self.predict_preprocessed_data_return_softmax(d, self.data_aug_params["mirror"], 1, False, 1,
                                                             self.data_aug_params['mirror_axes'], True, True, 2,
                                                             self.patch_size, True)
        pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        print("resampling to original spacing and nifti export...")
        save_segmentation_nifti_from_softmax(pred, output_file, properties, 3, None, None, None, softmax_ouput_file,
                                             None)
        print("done")

    def predict_preprocessed_data_return_softmax(self, data, do_mirroring, num_repeats, use_train_mode, batch_size,
                                                 mirror_axes, tiled, tile_in_z, step, min_size, use_gaussian):
        """
        Don't use this. If you need softmax output, use preprocess_predict_nifti and set softmax_output_file.
        :param data:
        :param do_mirroring:
        :param num_repeats:
        :param use_train_mode:
        :param batch_size:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param min_size:
        :param use_gaussian:
        :param use_temporal:
        :return:
        """
        assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        return self.network.predict_3D(data, do_mirroring, num_repeats, use_train_mode, batch_size, mirror_axes,
                                       tiled, tile_in_z, step, min_size, use_gaussian=use_gaussian,
                                       pad_border_mode=self.inference_pad_border_mode,
                                       pad_kwargs=self.inference_pad_kwargs)[2]

    def validate(self, do_mirroring=True, use_train_mode=False, tiled=True, step=2, save_softmax=True,
                 use_gaussian=True, compute_global_dice=True, override=True, validation_folder_name='validation'):
        """
        2018_12_05: I added global accumulation of TP, FP and FN for the validation in here. This is because I believe
        that selecting models is easier when computing the Dice globally instead of independently for each case and
        then averaging over cases. The Lung dataset in particular is very unstable because of the small size of the
        Lung Lesions. My theory is that even though the global Dice is different than the acutal target metric it is
        still a good enough substitute that allows us to get a lot more stable results when rerunning the same
        experiment twice. FYI: computer vision community uses the global jaccard for the evaluation of Cityscapes etc,
        not the per-image jaccard averaged over images.
        The reason I am accumulating TP/FP/FN here and not from the nifti files (which are used by our Evaluator) is
        that all predictions made here will have identical voxel spacing whereas voxel spacings in the nifti files
        will be different (which we could compensate for by using the volume per voxel but that would require the
        evaluator to understand spacings which is does not at this point)

        :param do_mirroring:
        :param use_train_mode:
        :param mirror_axes:
        :param tiled:
        :param tile_in_z:
        :param step:
        :param use_nifti:
        :param save_softmax:
        :param use_gaussian:
        :param use_temporal_models:
        :return:
        """
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)

        if do_mirroring:
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(4)
        results = []
        global_tp = OrderedDict()
        global_fp = OrderedDict()
        global_fn = OrderedDict()

        for k in self.dataset_val.keys():
            print(k)
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if override or (not isfile(join(output_folder, fname + ".nii.gz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_softmax(data[:-1], do_mirroring, 1,
                                                                             use_train_mode, 1, mirror_axes, tiled,
                                                                             True, step, self.patch_size,
                                                                             use_gaussian=use_gaussian)

                if compute_global_dice:
                    predicted_segmentation = softmax_pred.argmax(0)
                    gt_segmentation = data[-1]
                    labels = properties['classes']
                    labels = [int(i) for i in labels if i > 0]
                    for l in labels:
                        if l not in global_fn.keys():
                            global_fn[l] = 0
                        if l not in global_fp.keys():
                            global_fp[l] = 0
                        if l not in global_tp.keys():
                            global_tp[l] = 0
                        conf = ConfusionMatrix((predicted_segmentation == l).astype(int),
                                               (gt_segmentation == l).astype(int))
                        conf.compute()
                        global_fn[l] += conf.fn
                        global_fp[l] += conf.fp
                        global_tp[l] += conf.tp

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts 
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.9):  # *0.9 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")
                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, 3, None, None, None, softmax_fname, None),
                                                          )
                                                         )
                               )
                # save_segmentation_nifti_from_softmax(softmax_pred, join(output_folder, fname + ".nii.gz"),
                #                                               properties, 3, None, None,
                #                                               None,
                #                                               softmax_fname,
                #                                               None)

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        print("finished prediction, now evaluating...")

        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(tiled)),
                             json_author="Fabian",
                             json_task=task, num_threads=3)
        if compute_global_dice:
            global_dice = OrderedDict()
            all_labels = list(global_fn.keys())
            for l in all_labels:
                global_dice[int(l)] = float(2 * global_tp[l] / (2 * global_tp[l] + global_fn[l] + global_fp[l]))
            write_json(global_dice, join(output_folder, "global_dice.json"))

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Val glob dc per class:", str(global_dc_per_class))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
            lr = self.optimizer.param_groups[0]['lr']
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = lr
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()

        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        d_state = self.D.state_dict()
        for key in d_state.keys():
            d_state[key] = d_state[key].cpu()

        lr_sched_state_dct = None
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("UAGAN: saving checkpoint...")
        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'd_state': d_state,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics)},
            fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")

    def load_checkpoint(self, fname, train=True):
        print('UAGAN LOADER!')
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        self.load_checkpoint_ram(saved_model, train)

    def load_checkpoint_ram(self, saved_model, train=True):
        """
        used for if the checkpoint is already in ram
        :param saved_model:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys:
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)

        d_new_state = OrderedDict()
        d_curr_state_dict_keys = list(self.D.state_dict().keys())
        for k, value in saved_model['d_state'].items():
            key = k
            if key not in d_curr_state_dict_keys:
                key = key[7:]
            d_new_state[key] = value
        self.D.load_state_dict(d_new_state)

        self.epoch = saved_model['epoch']
        if train:
            optimizer_state_dict = saved_model['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
            if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.load_state_dict(saved_model['lr_scheduler_state_dict'])

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = saved_model['plot_stuff']

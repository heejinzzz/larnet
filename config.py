import torch
import os


class Config:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = "./Datasets"

        self.seed = 0
        self.lr = 2e-6
        self.epoch = 20
        self.num_workers = 1

        self.encode_text_len = None
        self.encode_audio_len = 96000

        self.modal = "bimodal"
        self.dataset = None
        self.num_classes = None
        self.max_uttrs_num = 60
        self.uttr_embedding_dim = 1280
        self.uttr_encoder_layers = 2
        self.uttrs_encoder_layers = 4
        self.uttr_encoder_nhead = 16
        self.uttrs_encoder_nhead = 16
        self.projection_expansion_factor = 4
        self.use_lar_attention = True
        self.lar_attention_pre_alpha = 0.1
        self.lar_attention_post_alpha = 0.2

        self.use_combine_loss = True
        self.sup_con_loss_temperature = None
        self.sup_con_loss_base_temperature = None
        self.sup_con_loss_weight = None

    def apply_args(self, args):
        if args.device is not None:
            self.device = args.device
        if args.dataset_path is not None:
            self.dataset_path = args.dataset_path
        if args.seed is not None:
            self.seed = args.seed
        if args.lr is not None:
            self.lr = args.lr
        if args.epoch is not None:
            self.epoch = args.epoch
        if args.num_workers is not None:
            self.num_workers = args.num_workers
        if args.modal is not None:
            self.modal = args.modal
        self.dataset = args.dataset
        if self.dataset == "meld":
            self.num_classes = 7
            self.encode_text_len = 256
            self.sup_con_loss_temperature = 0.25
            self.sup_con_loss_base_temperature = 0.25
            self.sup_con_loss_weight = 0.1
        elif self.dataset == "iemocap":
            self.num_classes = 6
            self.encode_text_len = 196
            self.sup_con_loss_temperature = 0.06
            self.sup_con_loss_base_temperature = 0.06
            self.sup_con_loss_weight = 1.0
        else:
            raise RuntimeError(f"Unknown Dataset: {self.dataset}")
        if args.max_uttrs_num is not None:
            self.max_uttrs_num = args.max_uttrs_num
        if args.uttr_embedding_dim is not None:
            self.uttr_embedding_dim = args.uttr_embedding_dim
        if args.uttr_encoder_layers is not None:
            self.uttr_encoder_layers = args.uttr_encoder_layers
        if args.uttrs_encoder_layers is not None:
            self.uttrs_encoder_layers = args.uttrs_encoder_layers
        if args.uttr_encoder_nhead is not None:
            self.uttr_encoder_nhead = args.uttr_encoder_nhead
        if args.uttrs_encoder_nhead is not None:
            self.uttrs_encoder_nhead = args.uttrs_encoder_nhead
        if args.projection_expansion_factor is not None:
            self.projection_expansion_factor = args.projection_expansion_factor
        self.use_lar_attention = not args.disable_lar_attention
        self.use_combine_loss = not args.disable_combine_loss

    def check(self):
        if not os.path.exists(self.dataset_path):
            raise RuntimeError(f"Non-existent path: {self.dataset_path}. Please place the downloaded datasets there")

    def __str__(self):
        return "Training Config\n-----------------------------------\n" +\
               f"device: {self.device}\n" +\
            f"dataset_path: {self.dataset_path}\n" +\
            f"random seed: {self.seed}\n" +\
            f"learning rate: {self.lr}\n" +\
            f"epoch: {self.epoch}\n" +\
            f"dataloader num_workers: {self.num_workers}\n" +\
            f"encode_text_len: {self.encode_text_len}\n" +\
            f"encode_audio_len: {self.encode_audio_len}\n" +\
            f"modal: {self.modal}\n" +\
            f"dataset: {self.dataset}\n" +\
            f"num_classes: {self.num_classes}\n" +\
            f"max_uttrs_num: {self.max_uttrs_num}\n" +\
            f"uttr_embedding_dim: {self.uttr_embedding_dim}\n" +\
            f"uttr_encoder_layers: {self.uttr_encoder_layers}\n" +\
            f"uttrs_encoder_layers: {self.uttrs_encoder_layers}\n" +\
            f"uttr_encoder_nhead: {self.uttr_encoder_nhead}\n" +\
            f"uttrs_encoder_nhead: {self.uttrs_encoder_nhead}\n" +\
            f"projection_expansion_factor: {self.projection_expansion_factor}\n" +\
            f"use_lar_attention: {self.use_lar_attention}\n" +\
            f"lar_attention_pre_alpha: {self.lar_attention_pre_alpha}\n" +\
            f"lar_attention_post_alpha: {self.lar_attention_post_alpha}\n" +\
            f"use_combine_loss: {self.use_combine_loss}\n" +\
            f"sup_con_loss_temperature: {self.sup_con_loss_temperature}\n" +\
            f"sup_con_loss_base_temperature: {self.sup_con_loss_base_temperature}\n" +\
            f"sup_con_loss_weight: {self.sup_con_loss_weight}\n"

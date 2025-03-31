from typing import List, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import verify_backbone_config_arguments
from ..auto import CONFIG_MAPPING
from .configuration_rf_detr_dinov2_with_registers import RFDetrDinov2WithRegistersConfig


logger = logging.get_logger(__name__)


class RFDetrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RFDetrModel`]. It is used to instantiate
    an RF DETR model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the RF DETR
    [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

    TODO: Add more details about the architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_queries (`int`, *optional*, defaults to 300):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
            [`RFDetrModel`] can detect in a single image. In case `two_stage` is set to `True`, we use
            `two_stage_num_proposals` instead.
        max_position_embeddings (`<fill_type>`, *optional*, defaults to 1024): <fill_docstring>
        encoder_layers (`int`, *optional*, defaults to 6):
            Number of encoder layers.
        encoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers.
        decoder_ffn_dim (`int`, *optional*, defaults to 1024):
            Dimension of the "intermediate" (often named feed-forward) layer in decoder.
        decoder_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        encoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        is_encoder_decoder (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        d_model (`int`, *optional*, defaults to 256):
            Dimension of the layers.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        init_xavier_std (`float`, *optional*, defaults to 1.0):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        return_intermediate (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        auxiliary_loss (`bool`, *optional*, defaults to `False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        position_embedding_type (`str`, *optional*, defaults to `"sine"`):
            Type of position embeddings to be used on top of the image features. One of `"sine"` or `"learned"`.
        backbone (`str`, *optional*, defaults to `"resnet50"`):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use pretrained weights for the backbone.
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        dilation (`bool`, *optional*, defaults to `False`):
            Whether to replace stride with dilation in the last convolutional block (DC5). Only supported when
            `use_timm_backbone` = `True`.
        num_feature_levels (`int`, *optional*, defaults to 4):
            The number of input feature levels.
        encoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the encoder.
        decoder_n_points (`int`, *optional*, defaults to 4):
            The number of sampled keys in each feature level for each attention head in the decoder.
        two_stage (`bool`, *optional*, defaults to `False`):
            Whether to apply a two-stage deformable DETR, where the region proposals are also generated by a variant of
            Deformable DETR, which are further fed into the decoder for iterative bounding box refinement.
        two_stage_num_proposals (`int`, *optional*, defaults to 300):
            The number of region proposals to be generated, in case `two_stage` is set to `True`.
        with_box_refine (`bool`, *optional*, defaults to `False`):
            Whether to apply iterative bounding box refinement, where each decoder layer refines the bounding boxes
            based on the predictions from the previous layer.
        class_cost (`float`, *optional*, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        bbox_cost (`float`, *optional*, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        giou_cost (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        mask_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the Focal loss in the panoptic segmentation loss.
        dice_loss_coefficient (`float`, *optional*, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        bbox_loss_coefficient (`float`, *optional*, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        giou_loss_coefficient (`float`, *optional*, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        eos_coefficient (`float`, *optional*, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        focal_alpha (`float`, *optional*, defaults to 0.25):
            Alpha parameter in the focal loss.
        disable_custom_kernels (`bool`, *optional*, defaults to `False`):
            Disable the use of custom CUDA and CPU kernels. This option is necessary for the ONNX export, as custom
            kernels are not supported by PyTorch ONNX export.
        out_feature_indexes (`List`, *optional*, defaults to `[2, 5, 8, 11]`): <fill_docstring>
        scale_factors (`List`, *optional*, defaults to `[1.0]`): <fill_docstring>
        layer_norm (`bool`, *optional*, defaults to `False`): <fill_docstring>
        projector_in_channels (`int`, *optional*, defaults to 256): <fill_docstring>
        projector_num_blocks (`int`, *optional*, defaults to 3): <fill_docstring>
        projector_survival_prob (`float`, *optional*, defaults to 1.0): <fill_docstring>
        projector_force_drop_last_n_features (`int`, *optional*, defaults to 0): <fill_docstring>

    Examples:

    ```python
    >>> from transformers import RFDetrConfig, RFDetrModel

    >>> # Initializing a Deformable DETR SenseTime/deformable-detr style configuration
    >>> configuration = RFDetrConfig()

    >>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
    >>> model = RFDetrModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rf_detr"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    def __init__(
        self,
        init_std=0.02,
        init_xavier_std=1.0,
        # backbone
        use_timm_backbone=False,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        backbone_kwargs=None,
        # RFDetrModel
        num_queries=300,
        # RFDetrEncoder
        encoder_layers=6,
        encoder_ffn_dim=1024,
        encoder_attention_heads=8,
        encoder_layerdrop=0.0,
        encoder_n_points=4,
        # RFDetrDecoder
        decoder_layers=3,
        d_model=256,
        attention_dropout=0.0,
        dropout=0.1,
        activation_function="relu",
        activation_dropout=0.0,
        decoder_self_attention_heads=8,
        decoder_cross_attention_heads=16,
        decoder_n_points=4,
        decoder_ffn_dim=2048,
        # LWDetr
        layer_norm: bool = True,
        ##
        auxiliary_loss=False,
        position_embedding_type="sine",
        dilation=False,
        two_stage=True,
        two_stage_num_proposals=300,
        with_box_refine=True,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        out_feature_indexes: List[int] = [2, 5, 8, 11],
        scale_factors: List[float] = [1.0],
        projector_in_channels: Optional[List[int]] = None,
        projector_num_blocks: int = 3,  # TODO rename
        projector_survival_prob: float = 1.0,
        projector_force_drop_last_n_features: int = 0,
        projector_activation_function: str = "silu",
        csp_hidden_expansion: float = 0.5,
        bottleneck_hidden_expansion: float = 0.5,
        batch_norm_eps: float = 1e-5,
        bbox_reparam: bool = True,
        is_encoder_decoder=True,
        num_groups=13,
        light_reference_point_refinement: bool = True,
        **kwargs,
    ):
        if backbone_config is None and backbone is None:
            logger.info(
                "`backbone_config` and `backbone` are `None`. Initializing the config with the default `RTDetr-ResNet` backbone."
            )
            backbone_config = RFDetrDinov2WithRegistersConfig(
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        verify_backbone_config_arguments(
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone=backbone,
            backbone_config=backbone_config,
            backbone_kwargs=backbone_kwargs,
        )

        self.use_timm_backbone = use_timm_backbone
        self.backbone_config = backbone_config
        self.num_queries = num_queries
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_self_attention_heads = decoder_self_attention_heads
        self.decoder_cross_attention_heads = decoder_cross_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.encoder_layerdrop = encoder_layerdrop
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.backbone_kwargs = backbone_kwargs
        self.dilation = dilation
        # deformable attributes
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.with_box_refine = with_box_refine
        if two_stage is True and with_box_refine is False:
            raise ValueError("If two_stage is True, with_box_refine must be True.")
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels

        self.scale_factors = [1.0] if scale_factors is None else scale_factors
        assert len(self.scale_factors) > 0, "scale_factors must be a list of at least one element"
        assert sorted(self.scale_factors, reverse=True) == self.scale_factors, "scale_factors must be reverse sorted"
        assert all(scale in [2.0, 1.0, 0.5, 0.25] for scale in self.scale_factors), (
            "scale_factors must be a consecutive list subset of [2.0, 1.0, 0.5, 0.25]"
        )

        self.num_feature_levels = len(scale_factors)
        self.layer_norm = layer_norm
        self.projector_in_channels = (
            projector_in_channels
            if projector_in_channels is not None
            else [backbone_config.hidden_size] * len(out_feature_indexes)
        )
        assert len(self.projector_in_channels) == len(out_feature_indexes), (
            "projector_in_channels must have the same length as out_feature_indexes"
        )
        self.projector_num_blocks = projector_num_blocks
        self.projector_survival_prob = projector_survival_prob
        self.projector_force_drop_last_n_features = projector_force_drop_last_n_features
        self.projector_activation_function = projector_activation_function
        self.csp_hidden_expansion = csp_hidden_expansion
        self.bottleneck_expansion = bottleneck_hidden_expansion
        self.batch_norm_eps = batch_norm_eps
        self.encoder_hidden_dim = backbone_config.hidden_size
        self.bbox_reparam = bbox_reparam
        self.num_groups = num_groups
        self.light_reference_point_refinement = light_reference_point_refinement
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model


__all__ = ["RFDetrConfig"]

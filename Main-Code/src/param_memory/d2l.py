"""接入原生 D2L 的加载、编码、编译、统一组合与完整安装。

模型依赖在构造后端时才导入。张量作为只读产物持有，调用方不能原地修改。
本模块不选择淘汰对象，不以文本读取或假编译器替代失败的参数后端。
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest.mock import patch

from .budget import check_token_limit
from .config import GenerationConfig, ModelConfig, fingerprint_file
from .schema import (
    BackendError,
    ChatMessages,
    CompileSpec,
    FactorRef,
    GenerationResult,
    TensorMetadata,
    TokenIds,
    compile_fingerprint,
    stable_json,
    text_hash,
)

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModuleLayout:
    """一个实际目标模块的输入/输出维度。"""

    name: str
    d_in: int
    d_out: int


@dataclass(frozen=True)
class FactorLayout:
    """实际 checkpoint 的层、rank、方向、精度、设备及当前预期宽度。"""

    modules: tuple[ModuleLayout, ...]
    n_layers: int
    rank: int
    width: int
    dtype: str
    device: str
    lora_alpha: float


@dataclass(frozen=True)
class RawFactors:
    """单块原始 A/B：[1, layers, rank, d_in/d_out]，不含共享项。"""

    tensors: Mapping[str, Mapping[str, Tensor]]
    ref: FactorRef


@dataclass(frozen=True)
class CombinedFactors:
    """完整读取 A/B：[1, layers, (N+1)*rank, d_in/d_out]，共享项一次。"""

    tensors: Mapping[str, Mapping[str, Tensor]]
    size_bytes: int


@dataclass(frozen=True)
class CompiledMemory:
    """一份当前或暂存状态的全部输入、原始因子、组合因子及元数据。"""

    specs: tuple[CompileSpec, ...]
    raw: Mapping[str, RawFactors]
    combined: CombinedFactors | None
    factor_refs: tuple[FactorRef, ...]

    @property
    def size_bytes(self) -> int:
        """返回这份状态实际持有的原始及组合张量字节。

        Returns:
            此状态引用的原始及组合张量总字节数，不含模型常驻内存。
        """
        return sum(ref.size_bytes for ref in self.factor_refs) + (
            self.combined.size_bytes if self.combined is not None else 0
        )


def empty_compiled() -> CompiledMemory:
    """创建纯基座状态，不编译空字符串、不分配共享项。

    Returns:
        无原始或组合因子的 CompiledMemory，不分配共享项。
    """
    return CompiledMemory(specs=(), raw=MappingProxyType({}), combined=None, factor_refs=())


def _readonly(tensors: dict) -> Mapping[str, Mapping[str, Tensor]]:
    """冻结两层模块映射；张量本身仍依赖调用方遵守只读契约。

    Args:
        tensors: 原生生成的两层模块与 A/B 张量字典。

    Returns:
        只读映射；张量不复制，调用方仍必须遵守不原地修改的契约。
    """
    return MappingProxyType({name: MappingProxyType(values) for name, values in tensors.items()})


class D2LBackend:
    """共享一个任务/管理基座，另持实际上下文编码器和冻结 hypernetwork。"""

    def __init__(self, config: ModelConfig) -> None:
        """加载本地原生模型并建立实际布局；失败抛 BackendError，不下载替代资源。

        Args:
            config: 已解析为绝对路径的模型配置；精度、设备和注意力实现均显式给定。

        Returns:
            None；实例持有冻结基座、上下文编码器、hypernetwork、tokenizer 和实际布局。

        Raises:
            BackendError: 资源、原生接口、CUDA 或张量布局不兼容。
        """
        self.config = config
        try:
            self._torch = import_module("torch")
            if config.device.startswith("cuda") and not self._torch.cuda.is_available():
                raise BackendError(f"model.device={config.device}，当前进程没有可用 CUDA 设备")
            source = config.d2l_root / "src"
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            native = import_module("ctx_to_lora.modeling.hypernet")
            if (
                Path(native.__file__).resolve()
                != (source / "ctx_to_lora/modeling/hypernet.py").resolve()
            ):
                raise BackendError("已导入的 ctx_to_lora 与 model.d2l_root 不一致")
            self._load_native(config, native)
            self._initialize_layout(native)
            self.reset()
        except (
            OSError,
            ImportError,
            ValueError,
            RuntimeError,
            KeyError,
            AttributeError,
            TypeError,
        ) as exc:
            if isinstance(exc, BackendError):
                raise
            raise BackendError(f"D2L 加载失败：{exc}") from exc

    def _load_native(self, config: ModelConfig, native: object) -> None:
        """复用原生构造器和严格权重加载，避免 from_state_dict 重建两次编码器。

        上游内部加载编码器时省略设备、精度和离线参数；只在构造期间修正该
        加载边界，使两个实际模型都服从已声明配置。退出作用域即还原上游函数。

        Args:
            config: 唯一模型配置。
            native: 已核对源码路径的原生 hypernet 模块。

        Returns:
            None；构造一次原生模型、严格装入权重，并记录实际资源指纹。

        Raises:
            BackendError: checkpoint 契约或声明的模型路径不匹配。

        Notes:
            模型、模板与指纹在同一装配边界完成，避免半加载实例被发布；原生构造只执行一次。
        """
        torch = self._torch
        state = torch.load(
            config.checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        self.checkpoint_model_id = state.pop("base_model_name_or_path")
        hyper_config = state.pop("hypernet_config")
        encoder_args = deepcopy(state.pop("ctx_encoder_args"))
        if hyper_config.extra_modules or not hyper_config.use_bias:
            raise BackendError("当前契约要求原始 LoRA 内容和一次共享项，不支持额外产物")
        if encoder_args.ctx_encoder_model_name_or_path not in (None, self.checkpoint_model_id):
            raise BackendError("当前配置没有独立编码器模型路径，不能下载另一个编码器")
        if encoder_args.quantize_ctx_encoder and not config.device.startswith("cuda"):
            raise BackendError("此 checkpoint 的量化上下文编码器需要 CUDA")
        encoder_args.ctx_encoder_model_name_or_path = str(config.base_model_path)
        loading = import_module("ctx_to_lora.model_loading")

        def load_local(model_name_or_path: str, **kwargs: object) -> object:
            """限定上游内部资源路径，并传入明确设备、精度、注意力实现。

            Args:
                model_name_or_path: 上游请求的模型路径，必须等于配置中的本地基座。
                **kwargs: 上游模型构造参数；设备、精度、注意力与离线限制在此明确传入。

            Returns:
                原生 get_model 创建的真实模型。

            Raises:
                BackendError: 上游尝试加载未声明的模型。
            """
            if Path(model_name_or_path).resolve() != config.base_model_path:
                raise BackendError(f"上游尝试加载未声明的模型：{model_name_or_path}")
            kwargs.update(
                device=config.device,
                dtype=getattr(torch, config.dtype),
                use_flash_attn=config.attention_implementation == "flash_attention_2",
            )
            kwargs["model_kwargs"] = {
                "local_files_only": True,
                "attn_implementation": config.attention_implementation,
            }
            return loading.get_model(model_name_or_path, **kwargs)

        base = load_local(
            str(config.base_model_path),
            train=False,
            requires_grad=False,
            peft_config=hyper_config.lora_config,
        )
        if base.config.hidden_size != hyper_config.base_hidden_size:
            raise BackendError("基座 hidden_size 与 checkpoint 不一致")
        with patch.object(native, "get_model", load_local):
            self.model = native.ModulatedPretrainedModel(
                base,
                hyper_config,
                encoder_args,
                use_sequence_packing=False,
            )
        self.model.hypernet.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.tokenizer = loading.get_tokenizer(
            str(config.base_model_path),
            tokenizer_kwargs={"local_files_only": True},
        )
        template = config.chat_template_path.read_text(encoding="utf-8")
        # 与上游 get_tokenizer 的定制模板规范化完全相同。
        self.tokenizer.chat_template = template.replace("    ", "").replace("\n", "")
        if not self.tokenizer.chat_template.strip():
            raise BackendError("明确选择的聊天模板为空")
        self._tokenize_context = import_module("ctx_to_lora.data.processing").tokenize_ctx_text
        self.template_fingerprint = text_hash(self.tokenizer.chat_template)
        self.context_capacity = int(base.config.max_position_embeddings)
        self.compiler_context_capacity = int(
            self.model.ctx_encoder.base_model.config.max_position_embeddings
        )
        self.resource_fingerprints = {
            "checkpoint": fingerprint_file(config.checkpoint_path),
            "chat_template": self.template_fingerprint,
            "base_metadata": {
                p.name: fingerprint_file(p) for p in sorted(config.base_model_path.glob("*.json"))
            },
            "base_weights": {
                p.name: fingerprint_file(p)
                for p in sorted(config.base_model_path.glob("*.safetensors"))
            },
            "native_sources": {
                str(p.relative_to(config.d2l_root)): fingerprint_file(p)
                for p in sorted((config.d2l_root / "src/ctx_to_lora").rglob("*.py"))
            },
        }
        self.compiler_fingerprint = text_hash(
            stable_json(
                {
                    "resources": self.resource_fingerprints,
                    "execution": config.model_dump(mode="json"),
                }
            )
        )

    def _initialize_layout(self, native: object) -> None:
        """从实际模型核对层、模块、A/B 方向、rank、scale、dtype 和设备。

        Args:
            native: 提供原生组合与安装接口的模块。

        Returns:
            None；建立真实 FactorLayout，并统计基座、编码器及 hypernetwork 常驻字节。

        Raises:
            BackendError: 层索引、模块维度或冻结状态不成立。

        Notes:
            布局与常驻资源取自同一批实际对象，避免形状声明和内存统计使用不同模型。
        """
        utils = import_module("ctx_to_lora.utils")
        hyper = self.model.hypernet
        layers = utils.get_layers(self.model.base_model)
        indices = tuple(int(index) for index in hyper.layer_indices)
        # 上游 apply_lora_to_layers 以实际层号索引产物，当前只接受完整连续层布局。
        if indices != tuple(range(len(layers))):
            raise BackendError("checkpoint 层索引不是与原生安装契约匹配的完整连续布局")
        d_in, d_out = utils.get_peft_in_out_features(self.model.base_model, self.model.peft_config)
        if (d_in, d_out) != hyper.config.feature_sizes:
            raise BackendError("实际目标模块的 A/B 维度与 checkpoint 不一致")
        parameter = next(hyper.parameters())
        self.device = parameter.device
        self.layout = FactorLayout(
            modules=tuple(
                ModuleLayout(name, d_in[name], d_out[name]) for name in hyper.target_modules
            ),
            n_layers=len(layers),
            rank=hyper.r,
            width=hyper.r,
            dtype=str(parameter.dtype),
            device=str(parameter.device),
            lora_alpha=float(self.model.peft_config.lora_alpha),
        )
        self._modules = tuple(
            info["module"]
            for layer in layers
            for info in utils.get_peft_modules(layer, self.model.peft_config)
        )
        self._combine_lora = native.combine_lora
        self._apply_lora = native.apply_lora_to_layers
        self.resident_bytes = {}
        for name, model in (
            ("base", self.model.base_model),
            ("context_encoder", self.model.ctx_encoder),
            ("hypernetwork", hyper),
        ):
            storages = {
                p.untyped_storage().data_ptr(): p.untyped_storage().nbytes()
                for p in (*model.parameters(), *model.buffers())
            }
            self.resident_bytes[name] = sum(storages.values())
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise BackendError("基座、编码器和 hypernetwork 必须全部冻结")
        logger.info("D2L loaded rank=%s layers=%s device=%s", hyper.r, len(layers), self.device)

    def encode_text(self, text: str) -> TokenIds:
        """返回原始 tokenizer IDs；不模板化、不截断、不执行模型。

        Args:
            text: 需计账的完整原始文本。

        Returns:
            不添加特殊 token 的 tokenizer IDs，不保留文本历史。
        """
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_context(self, text: str) -> TokenIds:
        """复用上游空 system/user/assistant 前缀路径，保持编译和计数完全一致。

        Args:
            text: 实际编译文本；上游 strip 仅作用于编译视图。

        Returns:
            与原生 tokenize_ctx_text 完全一致的完整编译输入 IDs。

        Raises:
            BackendError: 上下文模板或编码失败。
        """
        try:
            return tuple(self._tokenize_context({"context": [text]}, self.tokenizer)["ctx_ids"][0])
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise BackendError(f"D2L 上下文编码失败：{exc}") from exc

    def encode_messages(self, messages: ChatMessages) -> TokenIds:
        """用明确模板编码完整消息，带生成前缀；不保留前次 KV。

        Args:
            messages: 任务或管理模式的结构化消息。

        Returns:
            含明确聊天模板和 assistant 生成前缀的完整 IDs。

        Raises:
            BackendError: 消息或模板无法编码。
        """
        try:
            return tuple(
                self.tokenizer.apply_chat_template(
                    [message.model_dump(mode="json") for message in messages],
                    tokenize=True,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                )
            )
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise BackendError(f"聊天模板编码失败：{exc}") from exc

    def compile(self, spec: CompileSpec) -> RawFactors:
        """编译一份真实有界输入，返回 checkpoint 精度/设备的单块只读 A/B。

        Args:
            spec: 已校验的真实编译输入，含文本、IDs、指纹和来源版本。

        Returns:
            尚未加入共享项的单块 RawFactors，不安装参数。

        Raises:
            BackendError: 输入不一致、原生生成失败或产物不符合张量契约。
            BudgetExceeded: 输入超过实际上下文编码器容量。

        Notes:
            输入校验、原生调用与产物检查构成一个资源边界；失败时不向运行时交付半成品。
        """
        if spec.token_ids != self.encode_context(spec.text):
            raise BackendError("CompileSpec 的 token_ids 与实际上下文编码不一致")
        if spec.input_fingerprint != compile_fingerprint(spec.text, spec.token_ids):
            raise BackendError("CompileSpec 输入指纹不一致")
        if not spec.text.strip():
            raise BackendError("不能编译空白文本凑槽")
        check_token_limit(spec.token_ids, self.compiler_context_capacity, "上下文编码器输入")
        torch = self._torch
        try:
            with torch.inference_mode():
                ids = torch.tensor([spec.token_ids], dtype=torch.long, device=self.device)
                tensors, extra = self.model.generate_weights(ids, torch.ones_like(ids))
                if extra is not None or not isinstance(tensors, dict):
                    raise BackendError("generate_weights 返回了契约外产物")
                ref = FactorRef(
                    block_key=spec.block_key,
                    input_fingerprint=spec.input_fingerprint,
                    compiler_fingerprint=self.compiler_fingerprint,
                    content_width=self.layout.rank,
                    tensors=tuple(
                        TensorMetadata(
                            module=name,
                            matrix=matrix,
                            shape=tuple(tensor.shape),
                            dtype=str(tensor.dtype),
                            device=str(tensor.device),
                        )
                        for name, values in sorted(tensors.items())
                        for matrix, tensor in values.items()
                    ),
                    size_bytes=sum(
                        t.numel() * t.element_size()
                        for values in tensors.values()
                        for t in values.values()
                    ),
                )
                raw = RawFactors(_readonly(tensors), ref)
                self.validate_factors(raw, self.layout)
                self._synchronize()
                return raw
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise BackendError(f"D2L 编译 {spec.block_key} 失败：{exc}") from exc

    def compose(self, blocks: tuple[RawFactors, ...]) -> CombinedFactors | None:
        """统一组合所有原始内容块，共享项只加入一次；空序列走纯基座。

        Args:
            blocks: 全部活跃局部与全局的原始内容因子，按编译规格排序。

        Returns:
            加入一次共享项的完整组合产物；空集合返回 None，表示纯基座。

        Raises:
            BackendError: 张量或编译器身份不一致，或原生组合失败。

        Notes:
            所有内容同时组合，顺序及共享项只在此确定，不对单个块预先加入共享项。
        """
        if not blocks:
            return None
        for block in blocks:
            if not isinstance(block, RawFactors):
                raise BackendError("compose 只能接收尚未组合的原始内容因子")
            self.validate_factors(block, self.layout)
            if block.ref.compiler_fingerprint != self.compiler_fingerprint:
                raise BackendError("不能组合其它编译器产生的内容因子")
        torch = self._torch
        try:
            with torch.inference_mode():
                tensors = {
                    module.name: {
                        matrix: torch.cat(
                            [block.tensors[module.name][matrix] for block in blocks], dim=0
                        )
                        for matrix in ("A", "B")
                    }
                    for module in self.layout.modules
                }
                # 所有局部和全局原始因子同时组合；每个块不能各自先加一次共享项。
                combined = self._combine_lora(
                    tensors,
                    torch.tensor([len(blocks)], device=self.device),
                    lora_bias=self.model.hypernet.get_head_bias(),
                )
                result = CombinedFactors(
                    _readonly(combined),
                    sum(
                        tensor.numel() * tensor.element_size()
                        for values in combined.values()
                        for tensor in values.values()
                    ),
                )
                expected = replace(self.layout, width=(len(blocks) + 1) * self.layout.rank)
                self.validate_factors(result, expected)
                self._synchronize()
                return result
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            raise BackendError(f"D2L 组合失败：{exc}") from exc

    def validate_factors(
        self, factors: RawFactors | CombinedFactors, expected: FactorLayout
    ) -> None:
        """只读核对模块集合、A/B shape、dtype/device、有限值、宽度和实际字节。

        Args:
            factors: 实际原始或组合因子。
            expected: 从模型和本次内容块数派生的布局。

        Returns:
            None 表示 shape、方向、dtype、device、有限值、宽度与字节成立。

        Raises:
            BackendError: 实际张量与声明布局或 FactorRef 不一致。

        Notes:
            按模块同时核对实际张量和元数据，不通过转置、缩放或补零修复不匹配。
        """
        if set(factors.tensors) != {module.name for module in expected.modules}:
            raise BackendError("因子的目标模块集合不一致")
        actual_bytes = 0
        metadata = {}
        for module in expected.modules:
            values = factors.tensors[module.name]
            if set(values) != {"A", "B"}:
                raise BackendError(f"模块 {module.name} 必须且只能返回 A/B")
            for matrix, dimension in (("A", module.d_in), ("B", module.d_out)):
                tensor = values[matrix]
                shape = (1, expected.n_layers, expected.width, dimension)
                if tuple(tensor.shape) != shape:
                    raise BackendError(
                        f"{module.name}.{matrix} shape={tuple(tensor.shape)}，要求 {shape}"
                    )
                if str(tensor.dtype) != expected.dtype or str(tensor.device) != expected.device:
                    raise BackendError(f"{module.name}.{matrix} dtype/device 不一致")
                if tensor.requires_grad or not self._torch.isfinite(tensor).all().item():
                    raise BackendError(f"{module.name}.{matrix} 必须冻结且全部有限")
                actual_bytes += tensor.numel() * tensor.element_size()
                metadata[(module.name, matrix)] = TensorMetadata(
                    module=module.name,
                    matrix=matrix,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    device=str(tensor.device),
                )
        declared_bytes = (
            factors.ref.size_bytes if isinstance(factors, RawFactors) else factors.size_bytes
        )
        if actual_bytes != declared_bytes:
            raise BackendError(f"因子真实字节 {actual_bytes} 与元数据 {declared_bytes} 不一致")
        if isinstance(factors, RawFactors) and factors.ref.content_width != expected.rank:
            raise BackendError("原始因子内容宽度必须等于 checkpoint.rank")
        if isinstance(factors, RawFactors):
            declared = {(item.module, item.matrix): item for item in factors.ref.tensors}
            if declared != metadata or len(factors.ref.tensors) != len(metadata):
                raise BackendError("FactorRef 的模块/shape/dtype/device 与实际张量不一致")
            if expected.width != expected.rank:
                raise BackendError("原始单块不能使用组合宽度")

    def reset(self) -> None:
        """调用原生 reset 并核对残留绑定；重复调用安全，不删除 Runtime 持有的因子。

        Returns:
            None；清除当前模型的记忆 forward 与 generated_loras，不删除运行时产物。

        Raises:
            BackendError: 原生 reset 失败或仍残留绑定。
        """
        try:
            self.model.reset()
            if self.model.generated_loras is not None or any(
                getattr(module, "patched_forward", False) or module.forward != module.forward_orig
                for module in self._modules
            ):
                raise BackendError("原生 reset 后仍有记忆绑定")
        except (RuntimeError, AttributeError, TypeError) as exc:
            raise BackendError(f"D2L reset 失败：{exc}") from exc

    def install(self, factors: CombinedFactors | None) -> None:
        """先 reset 再一次性安装完整组合因子；不写基座权重，不重复组合。

        Args:
            factors: 已经统一组合的完整产物；None 表示纯基座。

        Returns:
            None；只完成绑定，发布前仍需 Runtime 执行 prefill。

        Raises:
            BackendError: reset、产物校验或原生安装失败。
        """
        self.reset()
        if factors is None:
            return
        if not isinstance(factors, CombinedFactors):
            raise BackendError("install 必须接收已组合产物或 None")
        if not factors.tensors:
            raise BackendError("非空组合产物必须包含实际张量")
        first = next(iter(factors.tensors.values())).get("A")
        if first is None or first.ndim != 4:
            raise BackendError("组合因子必须为四维 A/B")
        width = first.shape[2]
        self.validate_factors(factors, replace(self.layout, width=width))
        if width < self.layout.rank * 2 or width % self.layout.rank:
            raise BackendError("组合宽度必须对应至少一个内容块和一次共享项")
        try:
            self.model.patch_lora_forward()
            self._apply_lora(
                self.model.base_model,
                self.model.hypernet.layer_indices,
                factors.tensors,
                self._torch.ones(1, dtype=self._torch.int32, device=self.device),
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            raise BackendError(f"D2L 安装失败：{exc}") from exc

    def _synchronize(self) -> None:
        """等待本设备实际工作完成，使耗时和失败归属于当前操作。

        Returns:
            None；CUDA 操作已经完成，CPU 无需额外等待。
        """
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def prefill(self, input_ids: TokenIds) -> None:
        """对已安装状态执行真实前向；临时输出和 KV 不跨顶层调用保留。

        Args:
            input_ids: 已通过任务输入预算检查的完整工作上下文。

        Returns:
            None；执行真实前向及有限值检查，丢弃临时输出，不保存 KV。

        Raises:
            BackendError: 前向失败或最后位置 logits 非有限。
        """
        torch = self._torch
        try:
            with torch.inference_mode():
                ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                output = self.model.base_model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    return_dict=True,
                )
                if not torch.isfinite(output.logits[:, -1]).all().item():
                    raise BackendError("prefill 最后位置的 logits 包含非有限值")
                self._synchronize()
        except (RuntimeError, ValueError, TypeError) as exc:
            raise BackendError(f"D2L prefill 失败：{exc}") from exc

    def generate(self, input_ids: TokenIds, generation: GenerationConfig) -> GenerationResult:
        """直接调用已安装基座生成，只解码新 tokens，不跨调用保留 KV 或答案。

        Args:
            input_ids: 当前调用的完整输入，不含旧调用的 KV。
            generation: 此模式的输入/输出上限及采样选择。

        Returns:
            只含新生成 tokens 的 GenerationResult，保留原始解码和实际耗时。

        Raises:
            BudgetExceeded: 完整输入超过此模式上限。
            BackendError: 已安装基座的生成或解码失败。
        """
        check_token_limit(input_ids, generation.max_input_tokens, "生成输入")
        torch = self._torch
        self._synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                # 安装已完成组合，不能调用会再次 combine 的上游包装模型 generate。
                output = self.model.base_model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=True,
                    cache_implementation=None,
                    max_new_tokens=generation.max_new_tokens,
                    do_sample=generation.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new_ids = tuple(int(token) for token in output[0, len(input_ids) :].tolist())
                text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                self._synchronize()
            return GenerationResult(
                text=text,
                token_ids=new_ids,
                input_tokens=len(input_ids),
                output_tokens=len(new_ids),
                elapsed_seconds=time.perf_counter() - started,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise BackendError(f"D2L 生成失败：{exc}") from exc

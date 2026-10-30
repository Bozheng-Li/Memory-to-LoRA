"""读取唯一 YAML 配置并分阶段检查结构、数值和本地资源。

相对路径始终以 YAML 所在目录为基准。本模块不加载模型，不下载资源。
"""

import hashlib
import math
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError
from yaml.nodes import MappingNode

from .schema import SCHEMA_VERSION, ConfigError, FrozenModel, Nonempty


class ExperimentConfig(FrozenModel):
    """运行身份及统一种子。"""

    name: Nonempty
    seed: int


class ModelConfig(FrozenModel):
    """实际本地资源和明确选用的计算配置。"""

    d2l_root: Path
    checkpoint_path: Path
    base_model_path: Path
    chat_template_path: Path
    device: Nonempty
    dtype: Nonempty
    attention_implementation: Nonempty


class MemoryConfig(FrozenModel):
    """每次直接送入 D2L 的 chunk 上限，包含模板和特殊 token。"""

    chunk_tokens: int


class GenerationConfig(FrozenModel):
    """一种调用模式的显式输入/输出上限、解码与单次调用缓存设置。"""

    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    num_beams: int
    use_cache: bool
    cache_implementation: Nonempty | None


class GenerationProfiles(FrozenModel):
    """任务生成的显式解码配置。"""

    task: GenerationConfig


class PromptConfig(FrozenModel):
    """任务模式的固定指令模板。"""

    task_path: Path


class DataConfig(FrozenModel):
    """固定版本数据入口、官方评分源码和共同任务规则。"""

    manifest_path: Path
    selection: Nonempty
    official_scorer_path: Path
    protocol_instruction: Nonempty


class OutputConfig(FrozenModel):
    """运行产物父目录。"""

    directory: Path


class LoggingConfig(FrozenModel):
    """独立日志根目录及过程日志级别；逐条结果不受级别过滤。"""

    directory: Path
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class RunConfig(FrozenModel):
    """只进入入口、运行装配和评测编排的完整配置。"""

    schema_version: int
    experiment: ExperimentConfig
    model: ModelConfig
    memory: MemoryConfig
    generation: GenerationProfiles
    prompts: PromptConfig
    data: DataConfig
    output: OutputConfig
    logging: LoggingConfig


class UniqueKeySafeLoader(yaml.SafeLoader):
    """只在本读取器拒绝重复键、非字符串键及 YAML 合并。"""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, object]:
        """构造唯一字符串键映射；非法键抛含 YAML 位置的 ConfigError。

        Args:
            node: 当前 YAML mapping 节点。
            deep: 沿用 PyYAML 原生构造参数。

        Returns:
            键唯一的字符串字典，供严格配置模型校验。

        Raises:
            ConfigError: 非法节点抛 ConfigError，包含文件位置；不修改全局 SafeLoader 注册表。
        """
        mapping = {}
        for key_node, value_node in node.value:
            if key_node.tag != "tag:yaml.org,2002:str" or key_node.value == "<<":
                raise ConfigError(f"YAML 键必须为字符串且不能合并：{key_node.start_mark}")
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ConfigError(f"YAML 重复键 {key!r}：{key_node.start_mark}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_config(config_path: Path) -> RunConfig:
    """严格读取 YAML，只显式转换路径，返回尚待数值/资源校验的配置。

    Args:
        config_path: 现有 YAML 文件；相对路径以其父目录解析。

    Returns:
        RunConfig；数值不补默认，不合并环境变量或其它配置文件。

    Raises:
        ConfigError: 文件、编码、YAML、未知/缺失字段或严格类型错误。
    """
    try:
        path = config_path.resolve()
        with path.open(encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=UniqueKeySafeLoader)
        if not isinstance(data, dict):
            raise ConfigError("配置根节点必须是映射")
        # 由配置字段类型确定路径集合，避免另列一份可能遗漏的字段清单。
        for group, field in RunConfig.model_fields.items():
            cls = field.annotation
            if not isinstance(cls, type) or not issubclass(cls, FrozenModel):
                continue
            values = data.get(group)
            if not isinstance(values, dict):
                continue
            for name, child in cls.model_fields.items():
                if child.annotation is Path and name in values:
                    raw = values[name]
                    if not isinstance(raw, str) or not raw.strip():
                        raise ConfigError(f"{group}.{name}={raw!r}，要求非空路径字符串")
                    values[name] = (path.parent / raw).resolve()
        return RunConfig.model_validate(data)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"配置 {config_path}：{exc}") from exc


def validate_config(config: RunConfig) -> None:
    """检查范围、容量和支持的计算配置；成功不代表模型张量兼容。

    Args:
        config:  load_config 产生的类型化配置。

    Returns:
        None 表示静态配置约束成立，不代表模型已可运行。

    Raises:
        ConfigError: 冲突或不支持的配置抛 ConfigError；不修改输入，不加载模型。
    """
    if config.schema_version != SCHEMA_VERSION:
        raise ConfigError(f"schema_version={config.schema_version}，要求 {SCHEMA_VERSION}")
    if config.experiment.seed < 0:
        raise ConfigError(f"experiment.seed={config.experiment.seed}，要求非负整数")
    if config.logging.directory == config.output.directory:
        raise ConfigError("logging.directory 必须与 output.directory 分开")
    if config.memory.chunk_tokens <= 0:
        raise ConfigError(f"memory.chunk_tokens={config.memory.chunk_tokens}，要求正整数")
    profile = config.generation.task
    if profile.max_input_tokens <= 0 or profile.max_new_tokens <= 0:
        raise ConfigError(f"generation.task={profile}，输入和输出上限必须为正整数")
    for field in ("temperature", "repetition_penalty"):
        value = getattr(profile, field)
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"generation.task.{field}={value}，要求有限正数")
    if not math.isfinite(profile.top_p) or not 0 < profile.top_p <= 1:
        raise ConfigError(f"generation.task.top_p={profile.top_p}，要求 0 < top_p <= 1")
    for field, lower in (("top_k", 0), ("num_beams", 1)):
        if getattr(profile, field) < lower:
            raise ConfigError(f"generation.task.{field} 要求整数 >= {lower}")
    if not profile.use_cache and profile.cache_implementation is not None:
        raise ConfigError("generation.task.cache_implementation 在禁用缓存时必须为 null")
    model = config.model
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", model.device):
        raise ConfigError(f"model.device={model.device!r}，支持 cpu、cuda 或 cuda:设备号")
    if model.dtype not in {"float32", "float16", "bfloat16"}:
        raise ConfigError(f"model.dtype={model.dtype!r}，不支持该精度")
    if model.attention_implementation not in {"eager", "sdpa", "flash_attention_2"}:
        raise ConfigError(
            f"model.attention_implementation={model.attention_implementation!r}，不支持"
        )
    if model.device == "cpu" and model.attention_implementation == "flash_attention_2":
        raise ConfigError("model.attention_implementation=flash_attention_2 需要 CUDA")


def validate_resources(config: RunConfig) -> None:
    """检查明确资源和基座/tokenizer 文件；不创建目录，不触发下载。

    Args:
        config: 中路径已经解析。

    Returns:
        None 表示必需文件可访问；checkpoint 张量兼容性留给后端加载。

    Raises:
        ConfigError: 缺失或错误路径抛 ConfigError；不下载、不切换工作目录、不写文件。
    """
    directories = {
        "model.d2l_root": config.model.d2l_root,
        "model.base_model_path": config.model.base_model_path,
    }
    files = {
        "model.checkpoint_path": config.model.checkpoint_path,
        "model.chat_template_path": config.model.chat_template_path,
        "prompts.task_path": config.prompts.task_path,
        "data.manifest_path": config.data.manifest_path,
        "data.official_scorer_path": config.data.official_scorer_path,
        "model.base_config": config.model.base_model_path / "config.json",
        "model.tokenizer_config": config.model.base_model_path / "tokenizer_config.json",
        "model.tokenizer": config.model.base_model_path / "tokenizer.json",
        "model.d2l_source": config.model.d2l_root / "src/ctx_to_lora/modeling/hypernet.py",
    }
    for name, path in directories.items():
        if not path.is_dir():
            raise ConfigError(f"{name} 指向的目录不存在：{path}")
    for name, path in files.items():
        if not path.is_file():
            raise ConfigError(f"{name} 指向的文件不存在：{path}")
    if not any(config.model.base_model_path.glob("*.safetensors")):
        raise ConfigError(f"model.base_model_path 没有本地 safetensors 权重：{directories}")
    for name, directory in (
        ("output.directory", config.output.directory),
        ("logging.directory", config.logging.directory),
    ):
        for parent in (directory, *directory.parents):
            if parent.exists() and not parent.is_dir():
                raise ConfigError(f"{name} 的路径组件不是目录：{parent}")


def fingerprint_file(path: Path) -> str:
    """流式计算本地资源 SHA-256；只读文件，不把大权重整体复制进内存。

    Args:
        path: 本地资源文件。

    Returns:
        流式计算的 SHA-256 十六进制摘要。

    Raises:
        OSError: 文件无法读取。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

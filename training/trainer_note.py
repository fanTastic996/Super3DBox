# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers



# 训练器类，用于分布式数据并行（DDP）训练
class Trainer:
    """用于分布式数据并行（DDP）训练的通用训练器"""
    
    # 用于数值稳定性的小值
    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],           # 数据集配置
        model: Dict[str, Any],          # 模型配置
        logging: Dict[str, Any],        # 日志配置
        checkpoint: Dict[str, Any],    # 检查点配置
        max_epochs: int,               # 最大训练轮数
        mode: str = "train",           # 运行模式（train或val）
        device: str = "cuda",          # 训练设备（cuda或cpu）
        seed_value: int = 123,         # 随机种子
        val_epoch_freq: int = 1,       # 验证频率（每N轮验证一次）
        distributed: Dict[str, bool] = None,  # 分布式配置
        cuda: Dict[str, bool] = None,   # CUDA配置
        limit_train_batches: Optional[int] = None,  # 限制训练批次数
        limit_val_batches: Optional[int] = None,    # 限制验证批次数
        optim: Optional[Dict[str, Any]] = None,      # 优化器配置
        loss: Optional[Dict[str, Any]] = None,       # 损失函数配置
        env_variables: Optional[Dict[str, Any]] = None,  # 环境变量配置
        accum_steps: int = 1,          # 梯度累积步数
        **kwargs,
    ):
        """训练器初始化函数"""
        
        # 设置环境变量
        self._setup_env_variables(env_variables)
        # 初始化计时器
        self._setup_timers()

        # 存储Hydra配置
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # 存储超参数
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 训练进度（0.0-1.0），用于学习率调度器
        self.where = 0.0

        # 设置设备
        self._setup_device(device)
        # 设置分布式训练环境
        self._setup_torch_dist_and_backend(cuda, distributed)

        # 创建日志目录并配置日志
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        # 设置随机种子
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        # 确保分布式训练已初始化
        assert is_dist_avail_and_initialized(), "分布式训练需要先初始化"

        # 实例化组件（模型、损失函数等）
        self._setup_components()
        # 初始化数据加载器
        self._setup_dataloaders()

        # 将模型移动到指定设备
        self.model.to(self.device)
        # 初始化计时器
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # 如果是训练模式，构建优化器
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # 加载检查点（如果可用）
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:   
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # 使用DDP包装模型
        self._setup_ddp_distributed_training(distributed, device)
        
        # 同步所有进程
        dist.barrier()

    def _setup_timers(self):
        """初始化计时器"""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """设置环境变量"""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"环境变量:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """初始化分布式进程组并配置PyTorch后端"""
        # 如果CUDA可用，配置CUDA后端设置
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # 初始化分布式进程组
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        logging.info(f"从检查点恢复训练: {ckpt_path} (rank {self.rank})")

        # 加载检查点
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # 加载模型状态
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        if self.rank == 0:
            logging.info(f"模型状态已加载. 缺失键: {missing or '无'}. 意外键: {unexpected or '无'}.")

        # 如果是训练模式，加载优化器状态
        if "optimizer" in checkpoint and self.mode != "val":
            logging.info(f"加载优化器状态 (rank {self.rank})")
            self.optims.optimizer.load_state_dict(checkpoint["optimizer"])

        # 加载训练进度
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # 加载AMP scaler状态（如果可用）
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """设置训练设备"""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"不支持的设备: {device}")

    def _setup_components(self):
        """初始化训练组件：模型、损失函数、日志记录器等"""
        logging.info("初始化组件: 模型、损失函数、日志记录器等")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # 从配置实例化组件
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # 冻结指定的模型参数
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(f"[开始] 冻结模块: {self.optim_conf.frozen_module_names} (rank {self.distributed_rank})")
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(f"[完成] 冻结模块: {self.optim_conf.frozen_module_names} (rank {self.distributed_rank})")

        # 在rank 0上记录模型摘要
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"模型摘要已保存至 {model_summary_path}")

        logging.info("组件初始化完成")

    def _setup_dataloaders(self):
        """初始化训练和验证数据集的数据加载器"""
        self.train_dataset = None
        self.val_dataset = None

        # 如果是训练或验证模式，初始化验证数据集
        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        # 如果是训练模式，初始化训练数据集
        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """使用DistributedDataParallel（DDP）包装模型"""
        assert isinstance(self.model, torch.nn.Module)

        # DDP配置选项
        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        # 使用DDP包装模型
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        保存训练检查点
        
        参数:
            epoch: 当前训练轮数
            checkpoint_names: 检查点文件名列表（如["checkpoint_latest"]）
                             如果为None，则根据频率保存"checkpoint"和"checkpoint_{epoch}"
        """
        # 创建检查点目录
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        
        # 确定要保存的检查点名称
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        # 准备检查点内容
        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        # 如果只有一个优化器，简化结构
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        # 保存AMP scaler状态
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # 创建DDP检查点保存器
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        # 获取模型（如果是DDP模型，获取内部模块）
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        # 保存检查点
        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """获取需要记录的标量值键名"""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """训练/验证主入口"""
        assert self.mode in ["train", "val"], f"无效模式: {self.mode}"
        if self.mode == "train":
            self.run_train()  # 运行训练
            self.run_val()    # 训练后运行最终验证
        elif self.mode == "val":
            self.run_val()    # 只运行验证
        else:
            raise ValueError(f"无效模式: {self.mode}")

    def run_train(self):
        """训练主循环"""
        while self.epoch < self.max_epochs:
            # 设置每轮不同的随机种子
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            # 获取当前轮次的数据加载器
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            self.train_epoch(dataloader)  # 训练一个轮次
            
            # 保存检查点
            self.save_checkpoint(self.epoch)

            # 清理内存
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 按指定频率运行验证（跳过最后一轮）
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1  # 训练结束后回退一轮计数

    def run_val(self):
        """运行验证轮次"""
        if not self.val_dataset:
            logging.info("未配置验证数据集，跳过验证")
            return

        # 获取验证数据加载器
        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        self.val_epoch(dataloader)  # 运行验证轮次
        
        # 清理内存
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()  # 禁用梯度计算
    def val_epoch(self, val_loader):
        """验证轮次处理"""
        # 初始化性能监控器
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'  # 当前阶段：验证
        
        # 获取需要记录的损失名称
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        # 设置进度显示器
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="验证轮次: [{}]".format(self.epoch),
        )

        # 设置模型为评估模式
        self.model.eval()
        end = time.time()

        # 确定验证批次数量限制
        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        # 遍历验证数据
        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # 测量数据加载时间
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            # 预处理批次数据
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # 设置混合精度类型
            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"无效的AMP类型: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # 计算输出（不计算梯度）
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # 测量耗时
            batch_time.update(time.time() - end)
            end = time.time()

            # 更新总耗时
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            # 记录内存使用
            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # 按指定频率显示进度
            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def train_epoch(self, train_loader):        
        """训练轮次处理"""
        # 初始化性能监控器
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'  # 当前阶段：训练
        
        # 获取需要记录的损失名称
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        # 添加梯度监控
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        # 设置进度显示器
        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="训练轮次: [{}]".format(self.epoch),
        )

        # 设置模型为训练模式
        self.model.train()
        end = time.time()

        # 确定训练批次数量限制
        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        # 设置梯度裁剪
        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)

        # 遍历训练数据
        for data_iter, batch in enumerate(train_loader):

            if data_iter > limit_train_batches:
                break
            
            # 测量数据加载时间
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            # 预处理批次数据
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            # 为梯度累积分块批次数据
            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            # 在批次分块上运行训练步骤
            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # 更新学习率调度器
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"跳过调度器更新，因为训练已结束: {self.where} of [0,1]"
                )
                    
            # 记录调度器状态
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("优化器", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("优化器", "进度"),
                    self.where,
                    self.steps[phase],
                )

            # 梯度裁剪和异常梯度检测
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                # 记录梯度范数
                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # 优化器更新
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # 测量耗时
            batch_time.update(time.time() - end)
            end = time.time()
            # 更新总耗时
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            # 记录内存使用
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            # 按指定频率显示进度
            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        在批次分块上运行前向/反向传播
        每次反向传播累积梯度
        """        
        # 初始化优化器梯度
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        # 设置混合精度类型
        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"无效的AMP类型: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        # 处理每个分块
        for i, chunked_batch in enumerate(chunked_batches):
            # 对于非最后的分块，使用no_sync上下文减少同步次数
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                # 混合精度上下文
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    # 执行训练步骤
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )

                # 获取总损失并除以累积步数
                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                # 检查损失是否有限（非NaN或inf）
                if not math.isfinite(loss.item()):
                    error_msg = f"损失值异常: {loss.item()}, 尝试停止训练"
                    logging.error(error_msg)
                    return

                # 梯度累积：损失除以累积步数
                loss /= accum_steps
                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()
                # 更新损失记录器
                loss_meters[loss_key].update(loss.item(), batch_size)

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """应用数据增强：将原始批次与翻转版本拼接"""
        # 需要处理的张量键
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        # 需要处理的字符串键
        string_keys = ["seq_name"]
        
        # 处理张量：沿批次维度拼接原始和翻转版本
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        # 处理字符串：复制原始值
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        """批次数据预处理"""
        # 如果需要，应用批次重复增强
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # 标准化相机外参和点坐标
        normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

        # 用标准化值替换原始值
        batch["extrinsics"] = normalized_extrinsics
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        执行单步训练/验证
        包含前向传播、损失计算和日志记录
        
        返回:
            包含计算损失的字典
        """
        # 前向传播
        y_hat = model(images=batch["images"])
        
        # 损失计算
        loss_dict = self.loss(y_hat, batch)
        
        # 合并所有日志数据
        log_data = {**y_hat, **loss_dict, **batch}

        # 更新并记录标量值
        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        # 记录可视化结果
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        # 更新步骤计数
        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """更新平均值记录器并记录标量值到TensorBoard"""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]

        for key in keys_to_log:
            if key in data:
                # 获取值（如果是张量则转换为Python标量）
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                # 更新平均值记录器
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                # 在rank 0上记录到TensorBoard
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"标量值/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """记录图像/视频可视化到TensorBoard"""
        # 检查是否需要记录可视化
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        # 检查当前阶段是否有可视化配置
        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert len(keys_to_log) > 0, "需要指定要记录的可视化键"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in ["image", "video"], "目前只支持图像或视频记录"

            name = f"可视化/{phase}"

            # 创建可视化网格
            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # 取批次中的第一个样本
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)  # 限制值范围

            # 转换为NumPy数组
            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            # 记录到TensorBoard
            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )

# ===================== 工具函数 =====================

def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """为梯度累积将批次分割成更小的块"""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """检查数据是否是基本类型（str, int, float, bool）的序列"""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    递归分割数据结构的张量和序列
    
    参数:
        data: 要分割的数据结构（例如张量字典）
        chunk_id: 要获取的分块索引
        num_chunks: 总分割块数
    
    返回:
        原始数据结构的一个分块
    """
    # 处理张量或基本类型序列
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    # 处理字典
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    # 处理字符串（直接返回）
    elif isinstance(data, str):
        return data
    # 处理序列（递归处理每个元素）
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    # 其他类型直接返回
    else:
        return data
"""
Complete Examples: Using the Rewritten Components

This file contains fully working examples demonstrating:
1. Basic training setup
2. Distributed training (DDP)
3. Checkpoint resumption
4. Custom evaluation
5. End-to-end training pipeline
"""

# ============================================================================
# Example 1: Basic Training Setup
# ============================================================================

def example_basic_training():
    """Minimal working example for training."""
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.loader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    
    # Configuration
    train_config = TrainingConfig(
        n_steps=10000,
        learning_rate=1e-3,
        batch_size=32,
        eval_every=500,
        save_every=1000,
        log_every=10,
    )
    
    loader_config = DataLoaderConfig(
        batch_size=32,
        sequence_length=2048,
        buffer_size=1000,
    )
    # Initialize model and tokenizer
    from gpt_lib.model.gpt import DenseTransformer
    from gpt_lib.tokenizer import Tokenizer

    model = DenseTransformer()
    tokenizer = Tokenizer.from_pretrained("gpt2")
    
    # Initialize components
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
    )
    
    # Train
    trainer.train()


# ============================================================================
# Example 2: Download Dataset and Train
# ============================================================================

def example_with_download():
    """Example with dataset download before training."""
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    
    # Initialize ShardManager
    shard_manager = ShardManager(
        data_dir="./data",
        base_url="https://datasets.example.com/my-dataset"
    )
    
    # Download first 100 shards plus validation shard
    print("Downloading shards...")
    result = shard_manager.download(
        shard_indices=list(range(100)) + [9999],  # Last shard is validation
        num_workers=4,
        verbose=True,
    )
    print(f"Download complete: {result['successful']} successful, {result['failed']} failed")
    
    # List available shards
    shard_paths = shard_manager.list_shards()
    print(f"Available shards: {len(shard_paths)}")
    
    # Create dataloaders
    loader_config = DataLoaderConfig(batch_size=64)
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    # Train
    train_config = TrainingConfig(
        n_steps=50000,
        eval_every=1000,
        save_every=5000,
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
    )
    trainer.train()


# ============================================================================
# Example 3: Checkpoint Resumption
# ============================================================================

def example_resumption():
    """Example of saving and resuming from checkpoints."""
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    from pathlib import Path
    
    # Training configuration
    train_config = TrainingConfig(n_steps=100000)
    loader_config = DataLoaderConfig()
    
    # First training session
    print("=== Session 1: Training from scratch ===")
    
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
        checkpoint_dir="./checkpoints",
    )
    
    # Train for a bit, then interrupt
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        trainer.save_checkpoint(tag="interrupted")
    
    # Second training session (resumption)
    print("\n=== Session 2: Resuming from checkpoint ===")
    
    # Need fresh instances
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
        checkpoint_dir="./checkpoints",
    )
    
    # Load checkpoint and resume
    checkpoint_path = Path("./checkpoints/checkpoint_interrupted")
    trainer.load_checkpoint(checkpoint_path)
    
    print(f"Resuming training from step {trainer.state.global_step}")
    print(f"Best val loss so far: {trainer.state.best_val_loss:.6f}")
    
    # Continue training
    trainer.train()


# ============================================================================
# Example 4: Distributed Training (DDP)
# ============================================================================

def example_distributed_training():
    """Example with Distributed Data Parallel (DDP)."""
    import os
    import torch.distributed as dist
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    
    # Initialize DDP (assumes torch.distributed.launch or equivalent)
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Pin to local GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}/{world_size}] Initialized on device {device}")
    
    # Setup model with DDP wrapper
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    
    # Create dataloaders with DDP-aware sharding
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    loader_config = DataLoaderConfig(batch_size=32)
    
    train_loader = DataLoader(
        shard_manager,
        tokenizer,
        loader_config,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    val_loader = DataLoader(
        shard_manager,
        tokenizer,
        loader_config,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    
    # Training config with DDP info
    train_config = TrainingConfig(
        n_steps=10000,
        eval_every=500,
        dist_info={
            "RANK": rank,
            "WORLD_SIZE": world_size,
            "DEVICE_TYPE": "cuda",
        }
    )
    
    # Create trainer
    trainer = Trainer(
        model=model.module,  # Pass unwrapped model
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()


# ============================================================================
# Example 5: Custom Learning Rate Schedule
# ============================================================================

def example_custom_lr_schedule():
    """Example with custom learning rate schedule."""
    import math
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    
    # Define custom LR schedule (warmup + cosine decay)
    def lr_schedule(step: int, base_lr: float) -> float:
        """Warmup for 1000 steps, then cosine decay."""
        warmup_steps = 1000
        total_steps = 100000
        
        if step < warmup_steps:
            # Linear warmup
            return base_lr * (step / warmup_steps)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    # Create config with schedule callback
    train_config = TrainingConfig(
        n_steps=100000,
        learning_rate=1e-3,
        get_lr_schedule=lr_schedule,
    )
    
    # Rest of setup...
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    loader_config = DataLoaderConfig()
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
    )
    
    trainer.train()


# ============================================================================
# Example 6: Memory-Optimized Configuration
# ============================================================================

def example_memory_optimized():
    """Example tuned for GPUs with limited memory (e.g., 24GB)."""
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    
    # Reduce batch size and sequence length
    loader_config = DataLoaderConfig(
        batch_size=8,               # Small batch
        sequence_length=1024,       # Smaller sequence
        buffer_size=500,            # Smaller document buffer
        num_tokenizer_threads=2,    # Fewer threads
    )
    
    # Use gradient accumulation to maintain effective batch size
    train_config = TrainingConfig(
        n_steps=10000,
        batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch = 32
        eval_every=500,
    )
    
    # Setup
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
    )
    
    trainer.train()


# ============================================================================
# Example 7: Throughput-Optimized Configuration
# ============================================================================

def example_throughput_optimized():
    """Example tuned for maximum throughput (large GPU cluster)."""
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    
    # Large batch, large sequence, aggressive buffering
    loader_config = DataLoaderConfig(
        batch_size=256,             # Large batch
        sequence_length=4096,       # Large sequence
        buffer_size=5000,           # Large document buffer
        num_tokenizer_threads=16,   # Many threads
        tokenizer_batch_size=512,   # Large tokenizer batches
    )
    
    train_config = TrainingConfig(
        n_steps=100000,
        batch_size=256,
        gradient_accumulation_steps=1,  # No accumulation
        eval_every=1000,
        compile_model=True,  # Use torch.compile() for speedup
    )
    
    # Setup
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    train_loader = DataLoader(shard_manager, tokenizer, loader_config)
    val_loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
    )
    
    trainer.train()


# ============================================================================
# Example 8: Manual Iteration (Advanced)
# ============================================================================

def example_manual_iteration():
    """Example of manually iterating through shards without Trainer."""
    from gpt_lib.data.sharder import ShardManager, ShardIterationState
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    
    # Manual control over iteration
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    loader_config = DataLoaderConfig()
    loader = DataLoader(shard_manager, tokenizer, loader_config)
    
    # Train for exactly 1000 batches
    batch_count = 0
    for inputs, targets, state in loader.train(num_batches=1000):
        # Custom training step
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_count += 1
        
        # Custom logging
        if batch_count % 100 == 0:
            print(f"Batch {batch_count}: loss={loss.item():.6f}")
            # Save state for resumption
            if batch_count % 500 == 0:
                save_state({
                    "batch": batch_count,
                    "dataloader_state": state,
                    "model": model.state_dict(),
                })
    
    print(f"Training completed! Processed {batch_count} batches")
    
    # Run validation
    val_loss = 0.0
    val_count = 0
    for inputs, targets, _ in loader.val(num_batches=100):
        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits, targets)
            val_loss += loss.item()
            val_count += 1
    
    print(f"Validation loss: {val_loss / val_count:.6f}")


# ============================================================================
# Example 9: Integration with Weights & Biases
# ============================================================================

def example_with_wandb():
    """Example with W&B logging integration."""
    import wandb
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    from gpt_lib.utils.board import Board
    
    # Initialize W&B
    wandb.init(
        project="gpt-pretraining",
        name="experiment-1",
        config={
            "model": "gpt-large",
            "batch_size": 32,
            "learning_rate": 1e-3,
        }
    )
    
    # Create board with W&B backend
    board = Board(board_type="wandb")
    
    # Setup training
    train_config = TrainingConfig(
        n_steps=10000,
        eval_every=500,
    )
    
    shard_manager = ShardManager(data_dir="./data")
    shard_manager.list_shards()
    
    train_loader = DataLoader(shard_manager, tokenizer)
    val_loader = DataLoader(shard_manager, tokenizer)
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
        board=board,  # Pass board to trainer
    )
    
    # Train (all metrics automatically logged to W&B)
    trainer.train()
    
    wandb.finish()


# ============================================================================
# Example 10: Complete Production Setup
# ============================================================================

def example_production_setup():
    """
    Complete production-ready training setup with all components.
    """
    import torch
    import torch.distributed as dist
    from pathlib import Path
    
    from gpt_lib.data.sharder import ShardManager
    from gpt_lib.data.dataloader import DataLoader, DataLoaderConfig
    from gpt_lib.train.trainer import Trainer
    from gpt_lib.utils.schemas import TrainingConfig
    from gpt_lib.utils.board import Board
    
    # ================================================================
    # 1. Initialize Distributed Training
    # ================================================================
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # ================================================================
    # 2. Setup Logging Board
    # ================================================================
    board = Board(board_type="wandb") if rank == 0 else Board(board_type="dummy")
    
    # ================================================================
    # 3. Load Model and Tokenizer
    # ================================================================
    from gpt_lib.model.gpt import GPTModel
    from gpt_lib.tokenizer import Tokenizer
    
    model = GPTModel.from_pretrained("gpt-3-small").to(device)
    tokenizer = Tokenizer.from_pretrained("gpt-3")
    
    # Wrap with DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
        )
    
    # ================================================================
    # 4. Setup Data Pipeline
    # ================================================================
    shard_manager = ShardManager(
        data_dir="./data",
        base_url="https://datasets.example.com/openwebtext",
    )
    
    # Download dataset (rank 0 only)
    if rank == 0:
        shard_manager.download(num_workers=8)
    if world_size > 1:
        dist.barrier()  # Wait for downloads
    
    shard_manager.list_shards()
    
    # Create dataloaders with DDP sharding
    loader_config = DataLoaderConfig(
        batch_size=64,
        sequence_length=2048,
        buffer_size=2000,
    )
    
    train_loader = DataLoader(
        shard_manager, tokenizer, loader_config,
        ddp_rank=rank, ddp_world_size=world_size
    )
    val_loader = DataLoader(
        shard_manager, tokenizer, loader_config,
        ddp_rank=rank, ddp_world_size=world_size
    )
    
    # ================================================================
    # 5. Setup Training Configuration
    # ================================================================
    import math
    
    def cosine_lr_schedule(step, base_lr):
        warmup_steps = 1000
        total_steps = 100000
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    train_config = TrainingConfig(
        n_steps=100000,
        learning_rate=1e-3,
        batch_size=64,
        gradient_accumulation_steps=4,
        use_mixed_precision=True,
        compile_model=True,
        eval_every=1000,
        save_every=5000,
        log_every=10,
        get_lr_schedule=cosine_lr_schedule,
        dist_info={
            "rank": rank,
            "world_size": world_size,
            "device_type": "cuda",
        }
    )
    
    # ================================================================
    # 6. Create Trainer
    # ================================================================
    checkpoint_dir = Path("./checkpoints")
    
    trainer = Trainer(
        model=model if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=train_config,
        board=board,
        checkpoint_dir=checkpoint_dir,
    )
    
    # ================================================================
    # 7. Resume from Checkpoint if Available
    # ================================================================
    latest_checkpoint = checkpoint_dir / "checkpoint_latest"
    if latest_checkpoint.exists():
        if rank == 0:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
    
    # ================================================================
    # 8. Train
    # ================================================================
    if rank == 0:
        print("=" * 70)
        print(f"Starting training on {world_size} GPUs")
        print("=" * 70)
    
    trainer.train()
    
    # ================================================================
    # 9. Cleanup
    # ================================================================
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        print("Training completed successfully!")


if __name__ == "__main__":
    # Run one of the examples
    example_basic_training()
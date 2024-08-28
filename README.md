# LiteGen
 A light-weight and high-efficient training framework for accelerating diffusion tasks.

## 📜 About

LiteGen is a lightweight and high efficient training acceleration framework specifically designed for diffusion tasks, and has been applied and validated on video generation project [Vchitech-XL](). This framework integrates multiple training optimization techniques and offers a user-friendly interface, allowing researchers and developers to easily scale from single-GPU setups to multi-node, multi-GPU environments.

## ✨ Support Features

- vae:
  - DP vae
  - Sliced vae
  - vae.encode compile
- ema model:
  - sharded EMA (Exponential Moving Average)
- text encoder:
  - sharded text encoder
- distributed optimization
  - DDP
  - ZeRO1,2,3
  - Sequence Parallel (Ulysses implementation for Vchitect-XL Model) 
- memory optimization
  - Grad activation checkpointing
  - selective checkpointing

We also provide easy-to-use interfaces for common operations such as model loading and saving, etc. LiteGen allows users to focus on generative algorithm development and training without getting bogged down in implementation details.

## 🔨 Usage

Implementing LiteGen's optimizations involves two straightforward steps: 

1. **Configuration**: Adjust the relevant fields in your config file to enable desired optimizations. 
2. **Integration**: Utilize the api from the `LiteGen` instance in your codebase. These simple steps allow you seamlessly integrating optimizations into your existing workflow.

### Quick Start Guide

Follow these steps to integrate LiteGen into your project:

1. **Create LiteGen**

Create a `LiteGen` instance using your configuration file:

``` python
from litegen import LiteGen
gen = LiteGen(config)
```

2. **Initialize Components**

Use the `initialize` function to set up your training environment. This versatile function accepts various components, and returns the optimized version in the same order as the input arguments.

```python
model, optimizer, text_encoder, dataloader, vae_encode = gen.initialize(
    model,          # Your trainable model (only one is accepted)
    optimizer,      # Optimizer for the model
    text_encoder,   # Untrainable models (e.g., encoders in diffusion tasks)
    dataset,        # Your dataset
    vae.encode      # Computing functions (e.g., VAE encoding)
)
```

The two steps described above constitute the minimal code changes required to implement LiteGen's optimizations. This approach allows for quick integration while leveraging LiteGen's performance enhancements. 

In the following sections, we provide a detailed explanation of the specific optimizations LiteGen offers and how to configure the corresponding key-value pairs in the config file.

### Optimizations

#### DDP or ZeRO for the Trainable Model

LiteGen offers flexibility in choosing between Distributed Data Parallel (DDP) and different stages of ZeRO (Zero Redundancy Optimizer) for your trainable model. This choice is controlled by the `zero_degree` field in the configuration file:

- `zero_degree = 0 or None`: Uses DDP
- `zero_degree = 1`: Implements ZeRO Stage 1
- `zero_degree = 2`: Implements ZeRO Stage 2
- `zero_degree = 3`: Implements ZeRO Stage 3

When using ZeRO Stage 3, you can enable grouped ZeRO3 by setting `group_zero` to `True`. This option limits communication within a single node, potentially reducing inter-node communication overhead and enhancing training performance.

Example configuration:

```yaml
zero_degree: 3
group_zero: True
```

**Note**: While the `initialize` interface supports optimizing multiple models passed in any order, it only supports one trainable model. The function determines if a model is trainable by checking if any of its parameters have `requires_grad=True`. For all non-trainable models passed to the function, ensure you set `.requires_grad_(False)` beforehand.

#### Selective Activation Checkpointing

LiteGen incorporates activation checkpointing, a common optimization technique for reducing memory usage, and simplifies its usage. Furthermore, when sufficient memory is available, we allow for selective application of activation checkpointing to specific modules, thereby reducing performance overhead.

Example configuration:

```yaml
selective_ratio: 0.2    # Ratio of modules that do NOT use activation checkpointing. 
                        # 0: All blocks in the model use activation checkpointing.
                        # 1: No blocks in the model use activation checkpointing.
```

**Note:**

1. Activation checkpointing only applies to the trainable model.
2. Implement `get_fsdp_wrap_module_list()` in your model class to specify modules for checkpointing.
3. If not implemented, LiteGen automatically detects and applies checkpointing to repetitive module structures in the model (e.g., repeated transformer blocks in DiT models).

#### Activation Offload
🚧 Content is under construction.

#### Sequence Parallel
🚧 Content is under construction.


#### Sharded Untrainable Model
🚧 Content is under construction.

#### Computing Function Compile

🚧 Content is under construction.

#### Helpful Tools 

##### EMA Model
🚧 Content is under construction.

##### Checkpoint saving and loading
🚧 Content is under construction.


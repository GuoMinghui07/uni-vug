import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchvision import transforms
import time


# ==========================================
# 0. DDP 初始化与身份获取 (核心补充！)
# ==========================================
# 初始化进程组，"nccl" 是 NVIDIA 显卡之间通信最快的后端协议
dist.init_process_group(backend="nccl")

# 获取全局身份信息
rank = dist.get_rank()             # 当前进程的全局编号 (0, 1, 2, 3)
world_size = dist.get_world_size() # 总进程数 (4)

# 获取当前节点的局部 GPU 编号 (单机 4 卡情况下，local_rank 等于 rank)
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# 将当前进程绑定到对应的物理显卡上！极其重要！
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 只有主进程 (rank 0) 才打印日志，防止 4 张卡同时在终端刷屏
if rank == 0:
    print(f"🔥 DDP 初始化成功！总计 {world_size} 张显卡协同作战。")

# ==========================================
# 1. 禁用本地硬盘缓存
# ==========================================
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"

# 2. 定位 Fine-T2I 子集 (WebDataset .tar 格式)
url_pattern = "https://huggingface.co/datasets/ma-xu/fine-t2i/resolve/main/synthetic_original_prompt_square_resolution/train-*.tar"

# 3. 使用 webdataset 引擎进行流式加载
if rank == 0: print("🔗 正在连接 Fine-T2I 数据流...")
dataset = load_dataset("webdataset", data_files={"train": url_pattern}, streaming=True, split="train")

# 4. DDP 源头分流 (基于刚才获取的 rank 和 world_size)
dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

# 5. 超大内存洗牌池
dataset = dataset.shuffle(seed=42, buffer_size=1000)

# 6. 数据预处理
transform = transforms.Compose([
    transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((448, 448)),
    transforms.ToTensor(),
])

def preprocess(sample):
    return {"image": transform(sample["jpg"]), "prompt": sample["txt"]}

dataset = dataset.map(preprocess)

# 7. 组装 DataLoader
def custom_collate(batch):
    # batch 是一个列表，里面有 8 (或 64) 个字典
    # 我们只提取需要的 tensor 和文本，完全抛弃原始的 PIL Image
    images = torch.stack([item["image"] for item in batch])
    prompts = [item["prompt"] for item in batch]
    
    return {"image": images, "prompt": prompts}

dataloader = DataLoader(
    dataset,
    batch_size=64,             
    num_workers=16,            
    pin_memory=True,           
    prefetch_factor=4,         
    drop_last=True,
    collate_fn=custom_collate,
)

# ==========================================
# 🚀 极致性能测速模块
# ==========================================
if rank == 0:
    print("\n⏳ 开始预热 (跳过前 5 个 Batch 的网络连接和系统初始化开销)...")

# 1. 预热阶段 (Warm-up)
# PyTorch DataLoader 的前几个 batch 通常极慢，因为要拉起多进程和建立 HTTPS 连接
iterator = iter(dataloader)
for _ in range(5):
    batch = next(iterator)
    # 搭配 pin_memory=True 使用 non_blocking=True 可以实现异步拷贝！
    images = batch["image"].to(device, non_blocking=True)
    # 必须加上同步原语，否则计时器不准（因为 GPU 操作是异步的）
    torch.cuda.synchronize(device)

if rank == 0:
    print("🔥 预热完成，开始火力全开测速 (连续拉取 50 个 Batch)...")

# 2. 测速阶段
num_test_batches = 50
start_time = time.perf_counter()

for i in range(num_test_batches):
    batch = next(iterator)
    images = batch["image"].to(device, non_blocking=True)
    prompts = batch["prompt"]
    torch.cuda.synchronize(device) # 等待数据完全塞进显存

end_time = time.perf_counter()

# 3. 打印体检报告
if rank == 0:
    total_time = end_time - start_time
    time_per_batch = total_time / num_test_batches
    images_per_sec = (64 * num_test_batches) / total_time # 假设 batch_size=64
    
    print("\n" + "=" * 50)
    print("📊 DataLoader 极限吞吐量测试报告 (基于单卡视角):")
    print(f"⏱️ 测试总耗时: {total_time:.2f} 秒")
    print(f"📦 单个 Batch (64张 512x512 图) 备菜耗时: {time_per_batch:.4f} 秒")
    print(f"🚀 单卡拉取速度: {images_per_sec:.2f} 张图/秒")
    print(f"🌪️ 4卡总计吞吐量: {images_per_sec * world_size:.2f} 张图/秒")
    print("=" * 50)

# 最后别忘了优雅退出
import torch.distributed as dist
dist.destroy_process_group()
#!/bin/bash

# 定义日志文件路径
LOG_DIR="../../data/onion_train_data/train_results_2A-0.15"
LOG_FILE="$LOG_DIR/phantom2A-0.15_training-50-1.log"

# 确保日志目录存在，如果不存在则创建
mkdir -p "$LOG_DIR"
# # 定义日志文件路径
# LOG_FILE="../../onion_train_data/train_results_2A-0.15/phantom2A-0.15_training-50-1.log"
# 定义要运行的命令
COMMANDS=(
   # "python common_train.py --dataset phantom2A-0.15 --model Onion_input --randomnumseed 42 --device_num 0 --scheduler"
   # "python common_train.py --dataset phantom2A-0.15 --model Onion_input_softplus --randomnumseed 42 --device_num 0 --scheduler"
   # "python common_train.py --dataset phantom2A-0.15 --model Onion_PI_uptime --randomnumseed 42 --device_num 0 --scheduler"
   "python common_train.py --dataset phantom2A-0.15 --model ResOnion_input --randomnumseed 42 --device_num 4 --scheduler"
   "python common_train.py --dataset phantom2A-0.15 --model ResOnion_input_softplus --randomnumseed 42 --device_num 4 --scheduler"
   # "python common_train.py --dataset phantom2A-0.15 --model ResOnion_PI_uptime --randomnumseed 42  --device_num 0 --scheduler"
   "python common_train.py --dataset phantom2A-0.15 --model Onion_PI_uptime_softplus --randomnumseed 42 --device_num 4 --scheduler"
   "python common_train.py --dataset phantom2A-0.15 --model Onion_PI_uptime_softplus --addloss --randomnumseed 42 --device_num 4 --scheduler"
   "python common_train.py --dataset phantom2A-0.15 --model ResOnion_PI_uptime_softplus  --randomnumseed 42 --device_num 4 --scheduler"
   "python common_train.py --dataset phantom2A-0.15 --model ResOnion_PI_uptime_softplus --addloss --randomnumseed 42 --device_num 4 --scheduler"
)

# 遍历命令数组并依次执行每个命令
for cmd in "${COMMANDS[@]}"; do
    echo "Running command: $cmd" >> "$LOG_FILE" 2>&1
    eval "$cmd" >> "$LOG_FILE" 2>&1
    # 检查命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "An error occurred while running the command. Aborting." >> "$LOG_FILE" 2>&1
        exit 1
    fi
done

echo "All commands have been executed successfully." >> "$LOG_FILE" 2>&1
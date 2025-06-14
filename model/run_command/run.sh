#!/bin/bash

# 定义日志文件路径
LOG_FILE="../../onion_data/model_train_noregi_Nposi/up_training-50-4.log"
# 定义要运行的命令
COMMANDS=(
#    "python common_train.py --dataset EXP2A --model ResOnion_PI"
#    "python common_train.py --dataset EXP2A --model ResOnion_PI_up"
    "python common_train.py --dataset EXP2A --model ResOnion_PI_uptime_softplus --addloss"
    "python common_train.py --dataset EXP2A --model ResOnion_PI_uptime_softplus"
    "python common_train.py --dataset EXP2A --model Onion_PI_uptime_softplus --addloss"
    "python common_train.py --dataset EXP2A --model Onion_PI_uptime_softplus"

#    "python common_train.py --dataset phantom2A --model Onion_PI"
#    "python common_train.py --dataset phantom2A --model Onion_PI_up"
#    "python common_train.py --dataset phantom2A --model Onion_PI_uptime"


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
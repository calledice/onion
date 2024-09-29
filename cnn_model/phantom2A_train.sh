#!/bin/bash

# 定义日志文件路径
LOG_FILE="../../onion_data/model_train_0.0001/phantom2A_training-50-all.log"
# 定义要运行的命令
COMMANDS=(
#    "python common_train.py --dataset phantom2A --model Onion_input --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model Onion_input_softplus --randomnumseed 42 --device_num 1"
#    "python common_train.py --dataset phantom2A --model Onion_input_softplus --addloss --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model Onion_PI --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model Onion_PI_softplus --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model Onion_PI_softplus --addloss --lr 0.0001 --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model ResOnion_input --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model ResOnion_input_softplus --randomnumseed 42 --device_num 1"
#    "python common_train.py --dataset phantom2A --model ResOnion_input_softplus --addloss --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model ResOnion_PI --randomnumseed 42 --lr 0.0001 --device_num 1"
    "python common_train.py --dataset phantom2A --model ResOnion_PI_softplus --lr 0.0001 --randomnumseed 42 --device_num 1"
    "python common_train.py --dataset phantom2A --model ResOnion_PI_softplus --lr 0.0001 --addloss --randomnumseed 42 --device_num 1"

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
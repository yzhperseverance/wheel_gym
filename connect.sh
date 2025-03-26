#!/bin/bash

# 目标服务器信息
USER="yzh"
SERVER="10.192.40.124"
REMOTE_PATH="/home/yzh/"

# 本地路径
LOCAL_PATH="/home/yzh/人形机器人强化学习框架/wheel_gym/wheel_gym"

# 使用 rsync 传输并排除指定文件夹
rsync -avz --exclude='logs/test/*' --exclude='logs/flat_anymal_c' --exclude='.git*' --exclude='wheel_gym.egg-info' "$LOCAL_PATH" "$USER@$SERVER:$REMOTE_PATH"

if [ $? -eq 0 ]; then
  echo "文件传输成功！"
else
  echo "文件传输失败！"
fi

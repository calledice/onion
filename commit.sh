#!/bin/bash

# 两个参数: branch commit_message
# 用法: ./commit.sh branch "commit message"

branch="$1"
if [ $# -ne 2 ]; then
  echo "输入两个参数: branch commit_message"
  exit 1
fi

if [ "$1" != "gavin" ] || [ "$1" != "$branch"]; then
  echo "branch只能是$branch或者gavin"
  exit 1
fi

git checkout $branch
git add .
git commit -m "commit"
git push

if [ $? -ne 0 ]; then
  echo "git checkout $branch
git add .
git commit -m "commit"
>> git push
git checkout main
git pull
git merge $branch
解决冲突，重新提交
git push
git checkout $branch
git merge main
git push"
  echo "个人分支的push出问题了，解决问题之后重新提交。"
  exit 1
fi 
  
git checkout main
git pull
if [ $? -ne 0 ]; then
  echo "git checkout $branch
git add .
git commit -m "commit"
git push
git checkout main
>> git pull
git merge $branch
解决冲突，重新提交
git push
git checkout $branch
git merge main
git push"
  echo "pull main分支时候出问题，解决问题之后重新提交。"
  exit 1
fi 

git merge $branch
if [ $? -ne 0 ]; then
  echo "git checkout $branch
git add .
git commit -m "commit"
git push
git checkout main
git pull
git merge $branch
>> 解决冲突，重新提交
git push
git checkout $branch
git merge main
git push"
  echo "在main分支merge个人分支的提交时候出问题，解决问题之后重新merge。"
  exit 1
fi

git push
if [ $? -ne 0 ]; then
  echo "git checkout $branch
git add .
git commit -m "commit"
git push
git checkout main
git pull
git merge $branch
解决冲突，重新提交
>> git push
git checkout $branch
git merge main
git push"
  echo "在main分支merge个人分支完成之后，push出问题，解决问题之后重新push。"
  exit 1
fi

git checkout $branch
git merge main
if [ $? -ne 0 ]; then
  echo "git checkout $branch
git add .
git commit -m "commit"
git push
git checkout main
git pull
git merge $branch
解决冲突，重新提交
git push
git checkout $branch
>> git merge main
git push"
  echo "在个人分支merge个main支完时候出问题，解决问题之后重新merge。"
  exit 1
fi

git push
if [ $? -ne 0 ]; then
  echo "git checkout $branch
git add .
git commit -m "commit"
git push
git checkout main
git pull
git merge $branch
解决冲突，重新提交
git push
git checkout $branch
>> git merge main
git push"
  echo "在个人分支merge个main支完时候出问题，解决问题之后重新merge。"
  exit 1
fi
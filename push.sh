#!/bin/bash
# 检查是否有大于5MB的文件
MAX_SIZE=5242880
large_files=$(git diff --cached --name-only --diff-filter=AM | xargs ls -l 2>/dev/null | awk -v max="$MAX_SIZE" '$5 > max {print $NF}')

if [ -n "$large_files" ]; then
  echo "以下文件大于5MB，不能提交："
  echo "$large_files"
  exit 1
fi

git add .
git commit -m "update $(date +"%Y-%m-%d %H:%M")"
git push
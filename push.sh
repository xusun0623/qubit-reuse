while true; do
    git add .
    git commit -m "update $(date +"%Y-%m-%d %H:%M")"
    git push
    sleep 300  # 睡眠300秒（5分钟）
done
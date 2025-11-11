while true; do
    git add .
    git commit -m "update $(date +"%Y-%m-%d %H:%M")"
    git push
    git push github
    sleep 30
done
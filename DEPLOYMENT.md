# 本地：

```powershell
git add .
git commit -m "xxx"
git push
```

# 服务器

```bash
cd ~/apps/ai-customer-support/
git pull
sudo systemctl restart ai-customer-support
curl http://127.0.0.1:8000/health
```


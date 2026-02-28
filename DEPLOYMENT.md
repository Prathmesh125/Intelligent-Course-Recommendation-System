# Deploying NLPRec to Streamlit Community Cloud (FREE)

## Prerequisites
- GitHub account
- Your code pushed to a GitHub repository
- Streamlit Community Cloud account (free)

---

## Step 1: Prepare Your Repository

### ✅ Already Done:
- `requirements.txt` exists with all dependencies
- `.streamlit/config.toml` configured
- `.gitignore` set up to exclude models (they'll be generated automatically)
- App automatically builds models on first run

### Commit and Push:
```bash
git add .streamlit/ .gitignore
git commit -m "Add Streamlit Cloud deployment config"
git push origin main
```

---

## Step 2: Deploy on Streamlit Community Cloud

### A. Sign Up (if you haven't):
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Authorize Streamlit to access your repositories

### B. Deploy Your App:
1. Click **"New app"** button
2. Fill in:
   - **Repository**: `your-username/NLPRec` (or your repo name)
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **"Deploy"**

### C. Wait for Deployment:
- First deployment takes ~5-10 minutes
- Streamlit will:
  - Install all packages from `requirements.txt`
  - Download NLTK data automatically
  - Build TF-IDF models on first run
- You'll see build logs in real-time

---

## Step 3: Access Your App

Once deployed:
- You'll get a URL like: `https://your-app-name.streamlit.app`
- App will be live and accessible to anyone with the link
- Wakes up automatically when someone visits

---

## ⚠️ Important Notes About FREE Tier:

### Limitations:
- **Resources**: 1 GB RAM, 1 GB storage
- **Sleep Mode**: App sleeps after ~10 minutes of inactivity
  - Wakes up automatically when visited (takes ~10-30 seconds)
  - Models regenerate on wake if needed
- **Public**: Your code and app are publicly accessible
- **No custom domain**: Uses `.streamlit.app` subdomain

### "Lifetime Free" Reality:
✅ **Free as long as:**
- Streamlit Community Cloud exists
- You stay within resource limits
- Your repository is accessible
- You have an active GitHub account

❌ **Not Guaranteed Forever:**
- Streamlit could change their free tier policy
- Your account could be suspended for ToS violations
- Service could be discontinued (unlikely but possible)

**Practical Reality**: Streamlit Community Cloud has been free since 2021 and is backed by Snowflake (acquired Streamlit). It's very likely to remain free for personal/educational projects.

---

## Monitoring Your App

### Check Status:
1. Go to https://share.streamlit.io
2. View your app dashboard
3. See logs, metrics, and manage settings

### View Logs:
- Click on your app → "Manage app" → "Logs"
- See real-time console output
- Debug any issues

### Reboot App:
- Click "Reboot app" if something goes wrong
- Models will regenerate automatically

---

## Updating Your App

Any changes you push to GitHub will automatically redeploy:
```bash
# Make your changes
git add .
git commit -m "Update feature X"
git push origin main
```

Streamlit Cloud detects the push and redeploys automatically!

---

## Alternative FREE Hosting Options

If Streamlit Cloud doesn't meet your needs:

| Platform | Free Tier | Sleep Mode | Build Time |
|----------|-----------|------------|------------|
| **Streamlit Cloud** | ✅ Best for Streamlit | Yes (10 min) | ~5 min |
| **Render** | ✅ 750 hrs/month | Yes (15 min) | ~10 min |
| **Railway** | ✅ $5 credit/month | No | ~5 min |
| **Fly.io** | ✅ 3 VMs free | No | ~8 min |
| **Hugging Face Spaces** | ✅ Unlimited | No | ~7 min |

---

## Troubleshooting

### Models not loading:
- Check logs to see if vectorizer.py ran successfully
- Ensure `dataset/courses.csv` exists in repo

### Out of memory:
- Reduce dataset size if needed
- Use lighter models

### NLTK data not found:
- Already handled in `text_preprocessing.py`
- Runs `nltk.download()` automatically

### App crashes on startup:
- Check logs for Python errors
- Verify all dependencies in `requirements.txt`
- Reboot app from dashboard

---

## Next Steps

1. Deploy to Streamlit Cloud (easiest)
2. Share your app URL with others
3. Add custom analytics if needed
4. Consider paid hosting for production use (no sleep mode)

---

**Ready to deploy?** Run the commands in Step 1 and follow Step 2!

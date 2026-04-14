# JSON Data Agent — Deployment Guide
## Railway (backend) + Vercel (frontend) + GitHub (data)

---

## Folder structure

```
your-repo/
├── server.js          ← Railway backend
├── package.json
├── vercel.json        ← Vercel config
├── agent.html         ← Vercel frontend (what users see)
├── .gitignore
└── data/
    ├── 100009.json    ← your articles go here
    ├── 100010.json
    └── ...            ← add as many as you want
```

---

## Step 1 — Push to GitHub

```bash
# In your project folder:
git init
git add .
git commit -m "initial commit"
git branch -M main

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## Step 2 — Deploy backend on Railway

1. Go to [railway.app](https://railway.app) → **New Project**
2. Choose **Deploy from GitHub repo** → select your repo
3. Railway auto-detects `package.json` and runs `node server.js`
4. Go to your service → **Variables** tab → add:
   ```
   GROQ_API_KEY = gsk_your_key_here
   ```
5. Go to **Settings → Networking → Generate Domain**
6. Copy your Railway URL — looks like: `https://your-app.up.railway.app`

---

## Step 3 — Set your Railway URL in agent.html

Open `agent.html`, find line ~238:
```javascript
window.RAILWAY_SERVER_URL || "PASTE_YOUR_RAILWAY_URL_HERE"
```
Replace with your actual URL:
```javascript
window.RAILWAY_SERVER_URL || "https://your-app.up.railway.app"
```

Commit and push:
```bash
git add agent.html
git commit -m "set railway url"
git push
```

---

## Step 4 — Deploy frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import your GitHub repo
3. Vercel sees `vercel.json` and deploys `agent.html` automatically
4. Your public URL: `https://your-app.vercel.app` ← share this with users

---

## Step 5 — Update data (ongoing)

To add or update articles, just push new JSON files to the `data/` folder:

```bash
# Add new article
cp new-article.json data/100010.json

git add data/
git commit -m "add new articles"
git push
```

Railway auto-redeploys → server re-indexes → users see new data within ~30 seconds.

---

## How Railway auto-redeploys

Railway watches your GitHub repo. Every `git push` triggers a redeploy automatically.
No manual steps needed — push your JSON files and Railway handles the rest.

---

## Costs

| Service | Cost |
|---------|------|
| Railway | Free tier: $5 credit/month (enough for this) |
| Vercel | Free forever for static sites |
| Groq API | Free: 14,400 requests/day |
| GitHub | Free |

**Total: $0/month** for typical usage.

---

## Troubleshooting

**"Server offline" in the UI**
→ Check Railway dashboard — service may be sleeping (free tier sleeps after inactivity)
→ Click the service and wake it manually, or upgrade to hobby plan ($5/month) for always-on

**New JSON files not showing**
→ Make sure you pushed to the `main` branch
→ Check Railway deploy logs — should show "Indexed: yourfile.json"

**CORS error**
→ Make sure your Railway URL in agent.html does NOT have a trailing slash

**Railway URL not working**
→ Go to Railway → Settings → Networking → make sure a public domain is generated

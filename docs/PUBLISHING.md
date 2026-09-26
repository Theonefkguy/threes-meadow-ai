# GitHub Pages / 网站发布

The Pages workflow publishes only `dist/` on pushes to `main`. In repository Settings → Pages, choose **GitHub Actions** as source. No custom domain, API key or inference service is needed. The public URL is https://theonefkguy.github.io/threes-meadow-ai/.

工作流在 main 更新时先测试、校验模型和原始记录，再发布 dist/。训练数据仅作为仓库研究资料，不进入试玩网站。首次启用需在 Settings → Pages 选择 GitHub Actions。语言偏好和最高分仅保存于浏览器本地。

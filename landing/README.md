# localgate landing page

Static page for https://localgate.ghimireanjal.com.np. No build step: `index.html`, `config.js`, `img/`.

Preview locally: `python3 -m http.server -d landing 5500`, then open http://localhost:5500.
On localhost the page checks `backendUrl` from `config.js` (`/health/live`) and links to your
running gateway; on the public site it just points to the self-host steps.

## Deploy

Every host: no build command, **publish directory = `landing`**.

- **Vercel:** New Project → import the repo → Root Directory `landing`, Framework "Other", no build command.
- **Netlify:** New site → import repo → Base directory `landing`, Publish directory `landing` (or `.` with base set), build command empty.
- **Cloudflare Pages:** Create project → connect repo → Build command empty, Build output directory `landing`.

Then add the custom domain `localgate.ghimireanjal.com.np` in the host's dashboard and create the
CNAME record it gives you at your DNS provider.

`img/overview.png` and `img/agent-runs.png` are screenshots of the dashboard in the Sunrise theme (dark mode, 1440×900 at 1.5×), taken from a local install.

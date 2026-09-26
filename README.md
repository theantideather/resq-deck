# ResQ website

Static one page site for ResQ, the emergency response system built by Life ResQ Healthcare Systems LLP.

No build step, no framework, no dependencies to install. `index.html` is the whole site.

## Deploying on Netlify

1. In Netlify, choose **Add new site → Import an existing project** and pick this repository.
2. Leave the build command empty. Set the publish directory to `.` (already set in `netlify.toml`).
3. Deploy.

`netlify.toml` already sets security headers, cache rules and the 404 fallback, so nothing else needs configuring.

## Before the first real deploy

**Change the domain.** Four files hardcode `https://resqai.life`. If the site will live somewhere else, change it in all four or search engines will index the wrong canonical URL.

| File | What to change |
|---|---|
| `index.html` | `<link rel="canonical">`, the four `og:` and `twitter:` URL tags, and `url` / `logo` in the JSON-LD block |
| `robots.txt` | the `Sitemap:` line |
| `sitemap.xml` | the `<loc>` value |
| `netlify.toml` | nothing, unless you add a domain redirect |

**Check the Open Graph preview** once the domain resolves, using any link preview debugger. `og.png` is 1200x630 and already the right size.

## Files

| File | Purpose |
|---|---|
| `index.html` | The entire site: markup, styles and scripts in one file |
| `404.html` | Custom not found page, wired up in `netlify.toml` |
| `netlify.toml` | Publish settings, security headers, caching, redirects |
| `robots.txt`, `sitemap.xml` | Crawling and indexing |
| `site.webmanifest` | Installable web app metadata |
| `favicon.ico`, `favicon.svg`, `apple-touch-icon.png`, `icon-192.png`, `icon-512.png` | Icon set |
| `og.png` | Social share image |

## External resources

The page loads **one** thing from outside, allowed in the Content Security Policy in `netlify.toml`:

- **three.js r128** from cdnjs, for the 3D device on the hero. If it is blocked, the page still works: the 3D view falls back to a message pointing at the drawn elevation further down.

Fonts are self hosted in `/fonts` (Newsreader, IBM Plex Sans, IBM Plex Mono, latin subsets, 156 KB in total), so no request goes to Google and no visitor IP address is shared with them. If you want zero third party requests at all, vendor `three.min.js` into the repo and drop `cdnjs.cloudflare.com` from the CSP.

## The deck

`/deck` is the pitch deck as a self hosted page, built from the same slides as the Claude artifact. Arrow keys or click to move, `n` for speaker notes, `f` for full screen. The URL carries the slide number, so `/deck/#7` links straight to a slide. Printing the page gives one slide per page for a PDF.

Eight of the ten slides hold more content than fits a 1920x1080 slide. The viewer handles this by giving each slide the height it needs and zooming to fit, so nothing is ever cut off, but dense slides render smaller than sparse ones. The real fix is cutting copy rather than changing the viewer.

## Editing

Everything is in `index.html`, in this order: head and metadata, one `<style>` block, the markup, then one `<script>` block at the end. The script is split into three labelled parts: the session console, the council chamber, and the 3D device.

House rules for edits, which the current copy follows:

- No em dashes. Use a comma, a colon, a full stop or brackets.
- No claim that is not true today. No "patent pending" without a filing number, and no partner named as a partner until something is signed. Integrations that are being discussed say so.
- Every clinical line has to be correct. A dentist will check the nitrate line in the session replay, and it is right as written: nitrates are contraindicated within the PDE-5 inhibitor window.
- Sessions shown on the page are illustrations of designed behaviour. Keep them labelled that way.

## Status of what the page describes

The site states this plainly in its "Where we are" section, and that section should stay honest as things change. At the time of writing: the protocol library is in authoring, the council and avatar are in build, the desk unit is in hardware selection, clinic pilots are being recruited in Pune, and transport and hospital integrations are in discussion. The medical access card is in design.

## Before real patient data touches anything

The privacy section on the page covers the marketing site only, which collects nothing. The product itself will process patient data, and that needs a full privacy policy, a data processing agreement and a retention schedule reviewed by a lawyer under the DPDP Act 2023. The access card's break glass rules need the same review.

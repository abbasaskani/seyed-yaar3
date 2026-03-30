/* Seyd‑Yaar Service Worker — cache static assets, but ALWAYS refresh dynamic data (latest/ + runs/) */

const CACHE = "seydyaar-v0.6.1"; // bump this when you change SW

// Only STATIC assets here. Dynamic data must stay network-first.
const CORE = [
  "./",
  "./index.html",
  "./app.html",
  "./styles.css",
  "./home.js",
  "./app.js?v=docs-latest-v8",
  "./manifest.json",
  "./assets/logo.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const c = await caches.open(CACHE);
    await Promise.allSettled(CORE.map(async (url) => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (res && res.ok) await c.put(url, res.clone());
      } catch (_) {}
    }));
  })());
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => (k === CACHE ? null : caches.delete(k))));
    await self.clients.claim();
  })());
});

function isDynamic(url) {
  return (
    url.pathname.includes("/docs/latest/") ||
    url.pathname.includes("/latest/") ||
    url.pathname.includes("/runs/")
  );
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const acceptHeader = req.headers.get("accept") || "";
  if (req.mode === "navigate" || acceptHeader.includes("text/html")) {
    event.respondWith(
      fetch(req).catch(async () => (await caches.match("./app.html")) || (await caches.match("./index.html")) || new Response("Offline", { status: 503 }))
    );
    return;
  }

  if (isDynamic(url)) {
    event.respondWith(
      fetch(req, { cache: "no-store" })
        .then((res) => {
          if (res && res.ok) {
            const copy = res.clone();
            caches.open(CACHE).then((c) => c.put(req, copy));
          }
          return res;
        })
        .catch(async () => {
          const hit = await caches.match(req);
          return hit || new Response("Offline and not cached", { status: 503, statusText: "Service Unavailable" });
        })
    );
    return;
  }

  event.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      });
    })
  );
});

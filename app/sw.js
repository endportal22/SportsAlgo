/* Offline support: the app shell is cached; prediction data is fetched fresh when online
   and falls back to the last copy when offline. Bump VERSION when the shell changes. */
const VERSION = "sportsalgo-v7";
const SHELL = ["./", "index.html", "model.js", "manifest.webmanifest", "icons/icon-192.png", "icons/apple-touch-icon.png"];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  // network first (fresh predictions and app updates), cache as fallback
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(VERSION).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match(e.request).then((m) => m || caches.match("index.html"))));
});

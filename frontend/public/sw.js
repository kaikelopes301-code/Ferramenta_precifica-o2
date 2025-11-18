// Minimal no-op Service Worker to prevent 404 and allow safe registration.
// Scope: "/" (default). No caching; passthrough for all requests.
self.addEventListener('install', () => {
  // Activate immediately after installation
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  // Take control of uncontrolled clients without reload
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', () => {
  // No interception; let the network handle it
});

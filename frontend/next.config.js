// next.config.js

const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Evita que o Next infira a raiz errada quando há lockfiles no diretório pai
  outputFileTracingRoot: __dirname,

  // === OTIMIZAÇÕES DE PERFORMANCE ===

  reactStrictMode: true, // Detecta problemas com React mais cedo

  // Pacotes externos para o server (fora do bundle do Next)
  serverExternalPackages: ['@prisma/client', 'bcrypt'],

  // Imagens (otimização Next 15.x)
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost', port: '3000' },
      { protocol: 'https', hostname: 'precificacao.exemplo.com' },
    ],
    formats: ['image/webp', 'image/avif'],
    minimumCacheTTL: 60,
    dangerouslyAllowSVG: false,
    deviceSizes: [640, 750, 828, 1080, 1200, 1920],
    imageSizes: [16, 32, 48, 64, 96, 128, 256],
  },

  // Proxy para backend
  async rewrites() {
    return [
      {
        source: '/backend/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },

  // Otimizações de compilador
  compiler: {
    // Remove console.logs e propriedades do React em produção
    removeConsole: process.env.NODE_ENV === 'production',
    reactRemoveProperties: process.env.NODE_ENV === 'production',
  },

  // Headers de segurança
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-DNS-Prefetch-Control', value: 'on' },
          { key: 'Strict-Transport-Security', value: 'max-age=63072000; includeSubDomains; preload' },
          { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
        ],
      },
    ];
  },
};

// Exporta o config embrulhado no bundle analyzer
module.exports = withBundleAnalyzer(nextConfig);
/** @type {import('next').NextConfig} */
const nextConfig = {
	// Evita que o Next infira a raiz errada quando há lockfiles no diretório pai
	// e corrige rastreamento de dependências do lado do servidor.
	outputFileTracingRoot: __dirname,
	
	// === OTIMIZAÇÕES DE PERFORMANCE ===
  
	// React strict mode para detectar problemas early
	reactStrictMode: true,
	
	// SWC minifier removido - já é padrão no Next 15.x
	// swcMinify: true, // ❌ DEPRECADO
	
	// Server external packages (movido de experimental)
	serverExternalPackages: ['@prisma/client', 'bcrypt'],
	
	// Experimental optimizations
	experimental: {
		// CSS optimization desabilitada - problema com critters em Next 15.x
		// optimizeCss: true,
		// serverComponentsExternalPackages movido para serverExternalPackages
	},
	
	// Image optimization - Next 15.x format
	images: {
		// domains deprecado - usar remotePatterns
		remotePatterns: [
			{
				protocol: 'http',
				hostname: 'localhost',
				port: '3000',
			},
			{
				protocol: 'https',
				hostname: 'precificacao.exemplo.com',
			},
		],
		formats: ['image/webp', 'image/avif'],
		minimumCacheTTL: 60,
		dangerouslyAllowSVG: false,
		// Responsive image sizes
		deviceSizes: [640, 750, 828, 1080, 1200, 1920],
		imageSizes: [16, 32, 48, 64, 96, 128, 256],
	},
	
	// 🔄 PROXY CONFIGURATION - Solução A
	async rewrites() {
		return [
			{
				source: '/backend/:path*',
				destination: 'http://localhost:8000/:path*',
			},
		];
	},
	
	// Compiler optimizations
	compiler: {
		// Remove console.logs in production
		removeConsole: process.env.NODE_ENV === 'production',
		// React compiler optimizations
		reactRemoveProperties: process.env.NODE_ENV === 'production',
	},
	
	// Performance headers
	async headers() {
		return [
			{
				source: '/(.*)',
				headers: [
					{
						key: 'X-DNS-Prefetch-Control',
						value: 'on'
					},
					{
						key: 'Strict-Transport-Security',
						value: 'max-age=63072000; includeSubDomains; preload'
					},
					{
						key: 'X-Frame-Options',
						value: 'SAMEORIGIN'
					}
				],
			},
		];
	},
}

module.exports = nextConfig
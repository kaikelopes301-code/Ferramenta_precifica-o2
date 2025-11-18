import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import { Toaster } from "sonner"
import { ThemeProvider } from "next-themes"
import './globals.css'
import Metrics from "@/components/metrics"

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter'
})

export const metadata: Metadata = {
  title: 'Atlas Inovações - Ferramenta de Precificação',
  description: 'Sistema de precificação inteligente para equipamentos de limpeza com IA',
  keywords: ['precificação', 'equipamentos', 'IA', 'Atlas Inovações'],
  authors: [{ name: 'Atlas Inovações' }],
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' }
  ]
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="pt-br" suppressHydrationWarning className="scroll-smooth">
      <body className={`${inter.className} antialiased`} suppressHydrationWarning>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
          <Toaster 
            position="top-right"
            duration={3000}
            toastOptions={{
              style: {
                background: 'var(--color-card)',
                color: 'var(--color-card-foreground)',
                border: '1px solid var(--color-border)',
              },
            }}
          />
          {/* Métricas de Web Vitals (leve, sem mudar UI) */}
          <Metrics />
        </ThemeProvider>
      </body>
    </html>
  )
}

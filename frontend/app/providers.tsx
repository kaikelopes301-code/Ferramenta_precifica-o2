"use client"

import dynamic from "next/dynamic"
import { ThemeProvider } from "next-themes"
import type { ReactNode } from "react"

const Toaster = dynamic(() => import("sonner").then(m => m.Toaster), { ssr: false })

export default function Providers({ children }: { children: ReactNode }) {
  return (
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
    </ThemeProvider>
  )
}

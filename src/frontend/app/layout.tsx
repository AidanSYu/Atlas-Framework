import type { Metadata } from 'next'
import { Inter, Sora, IBM_Plex_Mono } from 'next/font/google'
import './globals.css'
import ToastContainer from '@/components/ToastContainer'
import { ThemeInitializerScript } from '@/components/ThemeInitializerScript'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
})

const sora = Sora({
  subsets: ['latin'],
  variable: '--font-display',
  display: 'swap',
})

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  variable: '--font-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Atlas - Research OS',
  description: 'Professional spatial research and intelligence platform',
  keywords: ['research', 'knowledge graph', 'AI', 'RAG', 'document analysis'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${sora.variable} ${ibmPlexMono.variable}`} suppressHydrationWarning>
      <body className={`${inter.className} bg-background text-foreground antialiased`} suppressHydrationWarning>
        <ThemeInitializerScript />
        {children}
        <ToastContainer />
      </body>
    </html>
  )
}

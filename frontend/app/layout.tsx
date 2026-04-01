import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import Link from 'next/link'
import './globals.css'
import { Providers } from './providers'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'RAG SaaS',
  description: 'Chat with your documents using AI',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col bg-gray-50">
        <Providers>
          <header className="border-b border-gray-200 bg-white shadow-sm">
            <nav className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
              <Link
                href="/"
                className="text-base font-bold text-gray-900 hover:text-blue-600 transition-colors"
              >
                RAG SaaS
              </Link>
              <div className="flex items-center gap-4 text-sm font-medium text-gray-600">
                <Link
                  href="/documents"
                  className="hover:text-blue-600 transition-colors"
                >
                  Documents
                </Link>
                <Link
                  href="/upload"
                  className="hover:text-blue-600 transition-colors"
                >
                  Upload
                </Link>
              </div>
            </nav>
          </header>
          <main className="flex flex-1 flex-col">{children}</main>
        </Providers>
      </body>
    </html>
  )
}

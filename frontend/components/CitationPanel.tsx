'use client'

import { Citation } from '@/lib/api'
import { Button } from '@/components/ui/button'

interface CitationPanelProps {
  citation: Citation | null
  onClose: () => void
}

export function CitationPanel({ citation, onClose }: CitationPanelProps) {
  if (!citation) return null

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col rounded-xl border border-gray-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3">
        <h3 className="text-sm font-semibold text-gray-800">Source {citation.source_n}</h3>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close citation panel">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="size-4"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
          </svg>
        </Button>
      </div>

      <div className="flex flex-col gap-3 overflow-y-auto p-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-gray-400">Document</p>
          <p className="mt-0.5 text-sm text-gray-800 break-words">{citation.doc_name}</p>
        </div>

        {citation.page !== null && (
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-gray-400">Page</p>
            <p className="mt-0.5 text-sm text-gray-800">{citation.page}</p>
          </div>
        )}

        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-gray-400">Excerpt</p>
          <blockquote className="mt-1 rounded-lg border-l-4 border-blue-300 bg-blue-50 px-3 py-2 text-sm italic text-gray-700 leading-relaxed">
            {citation.excerpt}
          </blockquote>
        </div>
      </div>
    </aside>
  )
}

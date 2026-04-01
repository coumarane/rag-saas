'use client'

import { Citation, Message } from '@/lib/api'

interface MessageBubbleProps {
  message: Message
  onCitationClick: (citation: Citation) => void
}

function parseContentWithCitations(
  content: string,
  citations: Citation[],
  onCitationClick: (citation: Citation) => void
): React.ReactNode[] {
  // Match [Source N] patterns
  const parts = content.split(/(\[Source \d+\])/g)

  return parts.map((part, index) => {
    const match = part.match(/^\[Source (\d+)\]$/)
    if (match) {
      const sourceN = parseInt(match[1], 10)
      const citation = citations.find((c) => c.source_n === sourceN)
      return (
        <button
          key={index}
          onClick={() => citation && onCitationClick(citation)}
          className={[
            'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold mx-0.5',
            'bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors',
            citation ? 'cursor-pointer' : 'cursor-default opacity-60',
          ].join(' ')}
          title={citation ? `Page ${citation.page ?? '?'}: ${citation.excerpt.slice(0, 80)}...` : undefined}
        >
          {part}
        </button>
      )
    }
    return <span key={index}>{part}</span>
  })
}

export function MessageBubble({ message, onCitationClick }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={['flex w-full', isUser ? 'justify-end' : 'justify-start'].join(' ')}>
      <div
        className={[
          'max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
          isUser
            ? 'bg-blue-600 text-white rounded-br-sm'
            : 'bg-gray-100 text-gray-800 rounded-bl-sm',
        ].join(' ')}
      >
        {isUser ? (
          <span>{message.content}</span>
        ) : (
          <span>
            {parseContentWithCitations(
              message.content,
              message.citations ?? [],
              onCitationClick
            )}
          </span>
        )}
      </div>
    </div>
  )
}

'use client'

import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { MessageBubble } from '@/components/MessageBubble'
import { CitationPanel } from '@/components/CitationPanel'
import { useChat } from '@/hooks/useChat'
import { Citation } from '@/lib/api'

interface ChatWindowProps {
  docId: string
  docName: string
}

function StreamingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-gray-100 px-4 py-3">
        <span className="size-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.3s]" />
        <span className="size-1.5 animate-bounce rounded-full bg-gray-400 [animation-delay:-0.15s]" />
        <span className="size-1.5 animate-bounce rounded-full bg-gray-400" />
      </div>
    </div>
  )
}

export function ChatWindow({ docId, docName }: ChatWindowProps) {
  const { messages, sendMessage, isStreaming, activeCitation, setActiveCitation } = useChat(docId)
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isStreaming])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    setInput('')
    sendMessage(trimmed)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleCitationClick = (citation: Citation) => {
    setActiveCitation(citation)
  }

  const showStreamingIndicator =
    isStreaming &&
    (messages.length === 0 ||
      messages[messages.length - 1]?.role !== 'assistant' ||
      messages[messages.length - 1]?.content === '')

  return (
    <div className="flex h-full gap-4">
      {/* Chat area */}
      <div className="flex flex-1 flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm">
        {/* Header */}
        <div className="border-b border-gray-100 px-5 py-3">
          <p className="text-xs text-gray-400">Chatting with</p>
          <p className="truncate text-sm font-semibold text-gray-800">{docName}</p>
        </div>

        {/* Messages */}
        <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-5 py-4">
          {messages.length === 0 && !isStreaming && (
            <div className="flex flex-1 items-center justify-center">
              <p className="text-sm text-gray-400">
                Ask a question about <span className="font-medium">{docName}</span>
              </p>
            </div>
          )}

          {messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              onCitationClick={handleCitationClick}
            />
          ))}

          {showStreamingIndicator && <StreamingIndicator />}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-100 px-4 py-3">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question... (Enter to send, Shift+Enter for new line)"
              rows={2}
              disabled={isStreaming}
              className="flex-1 resize-none rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-800 placeholder-gray-400 outline-none transition focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:opacity-50"
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="shrink-0"
            >
              {isStreaming ? 'Sending...' : 'Send'}
            </Button>
          </div>
        </div>
      </div>

      {/* Citation panel */}
      {activeCitation && (
        <CitationPanel
          citation={activeCitation}
          onClose={() => setActiveCitation(null)}
        />
      )}
    </div>
  )
}

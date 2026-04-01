'use client'

import { useState, useCallback, useRef } from 'react'
import { Citation, Message, createConversation, streamChat } from '@/lib/api'

export function useChat(docId: string) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [citations, setCitations] = useState<Citation[]>([])
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null)
  const conversationIdRef = useRef<string | null>(null)

  const sendMessage = useCallback(
    async (question: string) => {
      if (isStreaming) return

      // Append user message
      setMessages((prev) => [...prev, { role: 'user', content: question }])
      setIsStreaming(true)
      setCitations([])

      // Create conversation on first message
      if (!conversationIdRef.current) {
        try {
          const conversation = await createConversation(docId)
          conversationIdRef.current = conversation.id
        } catch (err) {
          console.error('Failed to create conversation:', err)
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: 'Failed to start conversation. Please try again.' },
          ])
          setIsStreaming(false)
          return
        }
      }

      // Append empty assistant message to stream into
      setMessages((prev) => [...prev, { role: 'assistant', content: '' }])

      let finalCitations: Citation[] = []

      try {
        await streamChat(
          question,
          docId,
          conversationIdRef.current,
          // onToken
          (token: string) => {
            setMessages((prev) => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              if (last && last.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  content: last.content + token,
                }
              }
              return updated
            })
          },
          // onCitations
          (receivedCitations: Citation[]) => {
            finalCitations = receivedCitations
            setCitations(receivedCitations)
          },
          // onDone
          () => {
            setMessages((prev) => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              if (last && last.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  citations: finalCitations,
                }
              }
              return updated
            })
            setIsStreaming(false)
          }
        )
      } catch (err) {
        console.error('Stream error:', err)
        setMessages((prev) => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: last.content || 'An error occurred. Please try again.',
            }
          }
          return updated
        })
        setIsStreaming(false)
      }
    },
    [docId, isStreaming]
  )

  return {
    messages,
    sendMessage,
    isStreaming,
    citations,
    activeCitation,
    setActiveCitation,
  }
}

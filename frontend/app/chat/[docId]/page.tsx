import Link from 'next/link'
import { ChatWindow } from '@/components/ChatWindow'
import { getDocument } from '@/lib/api'

interface ChatPageProps {
  params: Promise<{ docId: string }>
}

export default async function ChatPage({ params }: ChatPageProps) {
  const { docId } = await params

  let docName = 'Unknown Document'
  try {
    const doc = await getDocument(docId)
    docName = doc.file_name
  } catch {
    // Fall back to the docId if fetch fails
    docName = docId
  }

  return (
    <div className="flex h-[calc(100vh-57px)] flex-col px-6 py-4">
      <div className="mb-3 flex items-center gap-3">
        <Link
          href="/documents"
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 transition-colors"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="size-4"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18" />
          </svg>
          Documents
        </Link>
        <span className="text-gray-300">/</span>
        <span className="truncate text-sm font-medium text-gray-700">{docName}</span>
      </div>

      <div className="flex-1 overflow-hidden">
        <ChatWindow docId={docId} docName={docName} />
      </div>
    </div>
  )
}

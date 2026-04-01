'use client'

import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { useDocuments, useDeleteDocument } from '@/hooks/useDocuments'
import { DocumentResponse } from '@/lib/api'

const STATUS_STYLES: Record<DocumentResponse['status'], string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  processing: 'bg-blue-100 text-blue-800',
  ready: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
}

function StatusBadge({ status }: { status: DocumentResponse['status'] }) {
  return (
    <span
      className={[
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium capitalize',
        STATUS_STYLES[status],
      ].join(' ')}
    >
      {status}
    </span>
  )
}

function SkeletonRow() {
  return (
    <tr className="animate-pulse">
      {Array.from({ length: 6 }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <div className="h-4 rounded bg-gray-200" />
        </td>
      ))}
    </tr>
  )
}

export function DocumentList() {
  const router = useRouter()
  const { data: documents, isLoading } = useDocuments()
  const { mutate: deleteDocument, isPending: isDeleting } = useDeleteDocument()

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm('Are you sure you want to delete this document?')) {
      deleteDocument(id)
    }
  }

  const handleRowClick = (doc: DocumentResponse) => {
    if (doc.status === 'ready') {
      router.push(`/chat/${doc.id}`)
    }
  }

  return (
    <div className="w-full overflow-x-auto rounded-xl border border-gray-200 bg-white shadow-sm">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead className="bg-gray-50">
          <tr>
            {['Name', 'Type', 'Status', 'Chunks', 'Created', 'Actions'].map((col) => (
              <th
                key={col}
                className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {isLoading ? (
            <>
              <SkeletonRow />
              <SkeletonRow />
              <SkeletonRow />
            </>
          ) : !documents || documents.length === 0 ? (
            <tr>
              <td colSpan={6} className="px-4 py-12 text-center text-gray-400">
                No documents yet. Upload one to get started.
              </td>
            </tr>
          ) : (
            documents.map((doc) => (
              <tr
                key={doc.id}
                onClick={() => handleRowClick(doc)}
                className={[
                  'transition-colors',
                  doc.status === 'ready'
                    ? 'cursor-pointer hover:bg-blue-50'
                    : 'cursor-default',
                ].join(' ')}
              >
                <td className="px-4 py-3 font-medium text-gray-900 max-w-xs truncate">
                  {doc.file_name}
                </td>
                <td className="px-4 py-3 text-gray-500 uppercase text-xs">
                  {doc.file_type}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={doc.status} />
                </td>
                <td className="px-4 py-3 text-gray-500">
                  {doc.chunk_count ?? '—'}
                </td>
                <td className="px-4 py-3 text-gray-500">
                  {new Date(doc.created_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-3">
                  <Button
                    variant="destructive"
                    size="sm"
                    disabled={isDeleting}
                    onClick={(e) => handleDelete(doc.id, e)}
                  >
                    Delete
                  </Button>
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

import Link from 'next/link'
import { DocumentUpload } from '@/components/DocumentUpload'

export default function UploadPage() {
  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-12">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Document</h1>
        <p className="mt-1 text-sm text-gray-500">
          Upload a PDF or DOCX file to start chatting with it.
        </p>
      </div>

      <DocumentUpload />

      <p className="mt-6 text-center text-sm text-gray-500">
        Already uploaded?{' '}
        <Link href="/documents" className="font-medium text-blue-600 hover:underline">
          View your documents
        </Link>
      </p>
    </div>
  )
}

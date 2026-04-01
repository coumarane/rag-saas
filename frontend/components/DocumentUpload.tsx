'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { useUploadDocument } from '@/hooks/useDocuments'

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

export function DocumentUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [status, setStatus] = useState<UploadStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string>('')

  const { mutateAsync: uploadDocument } = useUploadDocument()

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0])
      setStatus('idle')
      setErrorMessage('')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false,
  })

  const handleUpload = async () => {
    if (!selectedFile) return
    setStatus('uploading')
    setErrorMessage('')
    try {
      await uploadDocument(selectedFile)
      setStatus('success')
      setSelectedFile(null)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed. Please try again.'
      setErrorMessage(message)
      setStatus('error')
    }
  }

  return (
    <div className="w-full max-w-lg mx-auto space-y-4">
      <div
        {...getRootProps()}
        className={[
          'border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors',
          isDragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100',
        ].join(' ')}
      >
        <input {...getInputProps()} />
        {isDragActive ? (
          <p className="text-blue-600 font-medium">Drop the file here...</p>
        ) : (
          <div className="space-y-2">
            <p className="text-gray-600 font-medium">
              Drag &amp; drop a file here, or click to select
            </p>
            <p className="text-sm text-gray-400">Supported formats: PDF, DOCX</p>
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="flex items-center gap-2 rounded-lg border border-gray-200 bg-white px-4 py-3">
          <svg
            className="size-5 text-gray-400 shrink-0"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
            />
          </svg>
          <span className="text-sm text-gray-700 truncate flex-1">{selectedFile.name}</span>
          <span className="text-xs text-gray-400 shrink-0">
            {(selectedFile.size / 1024).toFixed(1)} KB
          </span>
        </div>
      )}

      <Button
        onClick={handleUpload}
        disabled={!selectedFile || status === 'uploading'}
        className="w-full"
      >
        {status === 'uploading' ? 'Uploading...' : 'Upload Document'}
      </Button>

      {status === 'success' && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
          Document uploaded successfully!{' '}
          <Link href="/documents" className="font-medium underline underline-offset-2">
            View your documents
          </Link>
        </div>
      )}

      {status === 'error' && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {errorMessage || 'Upload failed. Please try again.'}
        </div>
      )}
    </div>
  )
}

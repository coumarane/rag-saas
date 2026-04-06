import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getDocuments,
  getDocument,
  uploadDocument,
  deleteDocument,
  DocumentResponse,
} from '@/lib/api'

const DOCUMENTS_KEY = ['documents']

export function useDocuments() {
  return useQuery<DocumentResponse[]>({
    queryKey: DOCUMENTS_KEY,
    queryFn: getDocuments,
    refetchInterval: (query) => {
      const documents = query.state.data
      const hasInFlightDocument = documents?.some(
        (doc) => doc.status === 'pending' || doc.status === 'processing'
      )
      return hasInFlightDocument ? 5000 : false
    },
  })
}

export function useDocument(id: string) {
  return useQuery<DocumentResponse>({
    queryKey: ['documents', id],
    queryFn: () => getDocument(id),
    enabled: !!id,
  })
}

export function useUploadDocument() {
  const queryClient = useQueryClient()
  return useMutation<DocumentResponse, Error, File>({
    mutationFn: (file: File) => uploadDocument(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY })
    },
  })
}

export function useDeleteDocument() {
  const queryClient = useQueryClient()
  return useMutation<void, Error, string>({
    mutationFn: (id: string) => deleteDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_KEY })
    },
  })
}

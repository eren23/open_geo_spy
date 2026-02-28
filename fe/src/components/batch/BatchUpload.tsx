import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BatchUploadProps {
  onUpload: (files: File[]) => void;
  loading: boolean;
}

interface FilePreview {
  file: File;
  url: string;
}

// ---------------------------------------------------------------------------
// BatchUpload -- Multi-file dropzone with previews
// ---------------------------------------------------------------------------

export default function BatchUpload({ onUpload, loading }: BatchUploadProps) {
  const [filePreviews, setFilePreviews] = useState<FilePreview[]>([]);

  // Revoke preview URLs on unmount
  useEffect(() => {
    return () => {
      for (const fp of filePreviews) {
        URL.revokeObjectURL(fp.url);
      }
    };
    // Only cleanup on unmount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      // Revoke old preview URLs
      for (const fp of filePreviews) {
        URL.revokeObjectURL(fp.url);
      }

      // Create new previews
      const previews = acceptedFiles.map((file) => ({
        file,
        url: URL.createObjectURL(file),
      }));
      setFilePreviews(previews);
    },
    [filePreviews],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    multiple: true,
    disabled: loading,
  });

  const handleSubmit = () => {
    if (filePreviews.length === 0) return;
    onUpload(filePreviews.map((fp) => fp.file));
  };

  const removeFile = (index: number) => {
    setFilePreviews((prev) => {
      const next = [...prev];
      URL.revokeObjectURL(next[index].url);
      next.splice(index, 1);
      return next;
    });
  };

  const clearAll = () => {
    for (const fp of filePreviews) {
      URL.revokeObjectURL(fp.url);
    }
    setFilePreviews([]);
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`
          relative rounded-xl border-2 border-dashed p-8
          text-center cursor-pointer transition-all duration-200
          ${
            isDragActive
              ? 'border-blue-500 bg-blue-50 scale-[1.01]'
              : loading
                ? 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-60'
                : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
        `}
      >
        <input {...getInputProps()} />

        {/* Upload icon */}
        <div className="mb-3 flex justify-center">
          <svg
            className="h-12 w-12 text-gray-400"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 3 3 0 013.834 3.855A3.75 3.75 0 0118 19.5H6.75z"
            />
          </svg>
        </div>

        <p className="text-sm font-medium text-gray-600">
          {isDragActive
            ? 'Drop images here...'
            : loading
              ? 'Processing batch...'
              : 'Drag & drop multiple images, or click to select'}
        </p>
        <p className="text-xs text-gray-400 mt-1">
          Supports JPEG, PNG, WebP -- select multiple files
        </p>

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center rounded-xl bg-white/60">
            <svg
              className="h-8 w-8 animate-spin text-blue-500"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
          </div>
        )}
      </div>

      {/* File preview list */}
      {filePreviews.length > 0 && (
        <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
          {/* Header */}
          <div className="px-4 py-2.5 border-b border-gray-200 flex items-center justify-between">
            <span className="text-sm font-semibold text-gray-900">
              {filePreviews.length} file{filePreviews.length !== 1 ? 's' : ''} selected
            </span>
            <button
              type="button"
              onClick={clearAll}
              disabled={loading}
              className="text-xs text-red-500 hover:text-red-700 disabled:opacity-50"
            >
              Clear all
            </button>
          </div>

          {/* File list */}
          <div className="divide-y divide-gray-100 max-h-64 overflow-y-auto">
            {filePreviews.map((fp, idx) => (
              <div
                key={`${fp.file.name}-${idx}`}
                className="flex items-center gap-3 px-4 py-2 hover:bg-gray-50"
              >
                {/* Thumbnail */}
                <img
                  src={fp.url}
                  alt={fp.file.name}
                  className="h-10 w-10 rounded object-cover flex-shrink-0 bg-gray-100"
                />

                {/* File info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-800 truncate">{fp.file.name}</p>
                  <p className="text-[11px] text-gray-400">
                    {(fp.file.size / 1024).toFixed(1)} KB
                  </p>
                </div>

                {/* Remove button */}
                <button
                  type="button"
                  onClick={() => removeFile(idx)}
                  disabled={loading}
                  className="flex-shrink-0 text-gray-400 hover:text-red-500 disabled:opacity-50 transition-colors"
                  title="Remove file"
                >
                  <svg
                    className="h-4 w-4"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    strokeWidth={2}
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>

          {/* Submit button */}
          <div className="px-4 py-3 border-t border-gray-200 bg-gray-50">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading || filePreviews.length === 0}
              className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Processing...' : `Upload ${filePreviews.length} image${filePreviews.length !== 1 ? 's' : ''}`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

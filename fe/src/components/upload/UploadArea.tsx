import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';

// ---------------------------------------------------------------------------
// UploadArea -- drag-drop file upload with image preview
// ---------------------------------------------------------------------------

interface UploadAreaProps {
  onUpload: (file: File) => void;
  loading: boolean;
}

export default function UploadArea({ onUpload, loading }: UploadAreaProps) {
  const [preview, setPreview] = useState<string | null>(null);

  // Revoke the object URL on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      // Create a preview URL
      if (preview) URL.revokeObjectURL(preview);
      setPreview(URL.createObjectURL(file));

      onUpload(file);
    },
    [onUpload, preview],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxFiles: 1,
    disabled: loading,
  });

  return (
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

      {/* Image preview */}
      {preview && (
        <img
          src={preview}
          alt="Upload preview"
          className="max-h-56 mx-auto rounded-lg mb-4 shadow-sm"
        />
      )}

      {/* Upload icon */}
      {!preview && (
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
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
            />
          </svg>
        </div>
      )}

      {/* Text */}
      <p className="text-sm font-medium text-gray-600">
        {isDragActive
          ? 'Drop image here...'
          : loading
            ? 'Processing image...'
            : 'Drag & drop an image, or click to select'}
      </p>
      <p className="text-xs text-gray-400 mt-1">
        Supports JPEG, PNG, WebP
      </p>

      {/* Loading spinner overlay */}
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
  );
}

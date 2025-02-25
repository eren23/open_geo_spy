import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onUploadSuccess: (result: any) => void;
  setLoading: (loading: boolean) => void;
  endpoint: string;
  title: string;
  description: string;
}

function FileUpload({ onUploadSuccess, setLoading, endpoint, title, description }: FileUploadProps) {
  const [locationHint, setLocationHint] = useState('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setLoading(true);
    const formData = new FormData();
    
    if (endpoint === '/api/locate') {
      formData.append('image', acceptedFiles[0]);
    } else {
      acceptedFiles.forEach(file => {
        formData.append('files', file);
      });
    }

    // Add location hint if provided
    if (locationHint.trim()) {
      formData.append('location', locationHint.trim());
    }

    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      onUploadSuccess(data);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setLoading(false);
    }
  }, [onUploadSuccess, setLoading, endpoint, locationHint]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png'],
      'video/*': ['.mp4']
    }
  });

  return (
    <div>
      <input
        type="text"
        placeholder="Enter location hint (optional)"
        value={locationHint}
        onChange={(e) => setLocationHint(e.target.value)}
        style={{
          width: '100%',
          padding: '8px',
          marginBottom: '10px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          fontSize: '14px'
        }}
      />
      
      <div
        {...getRootProps()}
        style={{
          border: '2px dashed #ccc',
          borderRadius: '8px',
          padding: '20px',
          textAlign: 'center',
          cursor: 'pointer',
          marginBottom: '20px',
          backgroundColor: isDragActive ? '#f0f0f0' : 'white'
        }}
      >
        <input {...getInputProps()} />
        <h3 style={{ margin: '0 0 10px 0' }}>{title}</h3>
        <p style={{ margin: 0 }}>{isDragActive ? 'Drop the files here...' : description}</p>
      </div>
    </div>
  );
}

export default FileUpload; 
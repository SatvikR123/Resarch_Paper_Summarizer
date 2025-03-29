import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { FiUpload, FiFile, FiAlertCircle } from 'react-icons/fi'

interface FileUploadProps {
  onFileUpload: (file: File) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const [error, setError] = useState<string | null>(null);
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    setError(null);
    if (acceptedFiles.length === 0) {
      return;
    }
    
    const file = acceptedFiles[0];
    if (file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB max
      setError('File size must be less than 10MB');
      return;
    }
    
    onFileUpload(file);
  }, [onFileUpload]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1
  });
  
  return (
    <div>
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center">
          <FiUpload className="text-4xl text-primary-500 mb-3" />
          <p className="text-lg font-medium text-gray-700">
            {isDragActive ? 'Drop your research paper here' : 'Drag & drop your research paper here'}
          </p>
          <p className="text-sm text-gray-500 mt-1">or click to browse files</p>
          <p className="text-xs text-gray-400 mt-2">PDF only, max 10MB</p>
        </div>
      </div>
      
      {error && (
        <div className="mt-2 flex items-center text-red-600">
          <FiAlertCircle className="mr-1" />
          <span className="text-sm">{error}</span>
        </div>
      )}
    </div>
  );
};

export default FileUpload; 
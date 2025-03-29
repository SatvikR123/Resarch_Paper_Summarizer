import React, { useState } from 'react';
import type { NextPage } from 'next';
import Layout from '../components/Layout';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';
import SummaryResult from '../components/SummaryResult';
import axios from 'axios';
import { FiUploadCloud, FiFileText } from 'react-icons/fi';

const Home: NextPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);
  const [images, setImages] = useState<Array<{ image: string; caption: string }>>([]);
  const [error, setError] = useState<string | null>(null);
  
  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    setSummary(null);
    setImages([]);
    setError(null);
  };
  
  const handleSubmit = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Change this to your API endpoint
      const response = await axios.post('http://localhost:8000/api/summarize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      const { summary, images } = response.data;
      setSummary(summary);
      setImages(images || []);
    } catch (err) {
      console.error('Error summarizing paper:', err);
      setError('An error occurred while summarizing the paper. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Layout>
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <FiFileText className="h-12 w-12 text-primary-600" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Research Paper Summarizer</h1>
          <p className="text-gray-600">
            Upload your research paper and get an AI-generated summary in seconds
          </p>
        </div>
        
        <FileUpload onFileUpload={handleFileUpload} />
        
        {file && !loading && !summary && (
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600 mb-3">
              Selected file: <span className="font-medium">{file.name}</span>
            </p>
            <button
              onClick={handleSubmit}
              className="button-primary inline-flex items-center"
            >
              <FiUploadCloud className="mr-2" />
              Summarize Paper
            </button>
          </div>
        )}
        
        {loading && (
          <div className="mt-8">
            <LoadingSpinner size="lg" text="Analyzing your research paper..." />
          </div>
        )}
        
        {error && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600">
            {error}
          </div>
        )}
        
        {summary && (
          <SummaryResult summary={summary} images={images} />
        )}
      </div>
    </Layout>
  );
};

export default Home; 
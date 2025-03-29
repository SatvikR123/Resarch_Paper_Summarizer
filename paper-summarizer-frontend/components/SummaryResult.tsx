import React from 'react';
import { FiCopy, FiDownload, FiCheckCircle } from 'react-icons/fi';

interface SummaryResultProps {
  summary: string;
  images?: Array<{ image: string; caption: string }>;
}

const SummaryResult: React.FC<SummaryResultProps> = ({ summary, images = [] }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([summary], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = 'summary.txt';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="bg-white rounded-xl shadow-md p-6 mt-8">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800">Summary</h2>
        <div className="flex space-x-2">
          <button
            onClick={handleCopy}
            className="flex items-center text-primary-600 hover:text-primary-700 transition-colors"
            title="Copy to clipboard"
          >
            {copied ? <FiCheckCircle className="h-5 w-5" /> : <FiCopy className="h-5 w-5" />}
            <span className="ml-1 text-sm">{copied ? 'Copied!' : 'Copy'}</span>
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center text-primary-600 hover:text-primary-700 transition-colors"
            title="Download as text file"
          >
            <FiDownload className="h-5 w-5" />
            <span className="ml-1 text-sm">Download</span>
          </button>
        </div>
      </div>
      <div className="prose prose-primary max-w-none">
        <p className="text-gray-700 whitespace-pre-line">{summary}</p>
      </div>

      {images.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Extracted Figures</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {images.map((img, index) => (
              <div key={index} className="border rounded-lg overflow-hidden">
                <img src={img.image} alt={img.caption} className="w-full h-auto" />
                <div className="p-3 bg-gray-50">
                  <p className="text-sm text-gray-600">{img.caption}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SummaryResult; 
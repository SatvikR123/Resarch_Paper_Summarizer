import type { NextPage } from 'next';
import Layout from '../components/Layout';
import { FiFileText, FiZap, FiLock, FiCpu } from 'react-icons/fi';

const About: NextPage = () => {
  return (
    <Layout>
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-gray-900 mb-3">About Research Paper Summarizer</h1>
          <p className="text-gray-600 text-lg">
            Powered by state-of-the-art AI to make research more accessible
          </p>
        </div>

        <div className="prose prose-primary max-w-none">
          <p>
            Research Paper Summarizer is an advanced tool designed to help researchers, students, and professionals quickly extract the key information from scientific papers. Our system uses a fine-tuned BART model specifically trained on academic research papers to generate concise, accurate summaries.
          </p>

          <h2 className="mt-8">How It Works</h2>
          <p>
            Our system uses a BART Large model that has been fine-tuned on a dataset of research papers and their summaries. We've implemented LoRA (Low-Rank Adaptation) to efficiently adapt the pre-trained model to the specific task of summarizing academic content, preserving the model's general knowledge while optimizing it for scientific text.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
            <div className="paper-card p-6">
              <FiZap className="h-8 w-8 text-primary-600 mb-3" />
              <h3 className="text-xl font-bold text-gray-800 mb-2">Fast Processing</h3>
              <p className="text-gray-600">
                Upload your paper and get a summary in seconds, saving you valuable research time.
              </p>
            </div>

            <div className="paper-card p-6">
              <FiCpu className="h-8 w-8 text-primary-600 mb-3" />
              <h3 className="text-xl font-bold text-gray-800 mb-2">Advanced AI</h3>
              <p className="text-gray-600">
                Powered by state-of-the-art NLP models specifically trained on scientific literature.
              </p>
            </div>

            <div className="paper-card p-6">
              <FiFileText className="h-8 w-8 text-primary-600 mb-3" />
              <h3 className="text-xl font-bold text-gray-800 mb-2">Comprehensive Analysis</h3>
              <p className="text-gray-600">
                Extracts key findings, methodology, and conclusions from complex research papers.
              </p>
            </div>

            <div className="paper-card p-6">
              <FiLock className="h-8 w-8 text-primary-600 mb-3" />
              <h3 className="text-xl font-bold text-gray-800 mb-2">Privacy Focused</h3>
              <p className="text-gray-600">
                Your uploaded papers are processed securely and not stored after summarization.
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default About; 
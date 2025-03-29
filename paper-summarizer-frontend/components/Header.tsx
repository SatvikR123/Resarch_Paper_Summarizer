import React from 'react'
import Link from 'next/link'
import { FiHome, FiInfo, FiFileText } from 'react-icons/fi'

const Header: React.FC = () => {
  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/" className="flex items-center">
                <FiFileText className="h-8 w-8 text-primary-600" />
                <span className="ml-2 text-xl font-bold text-gray-900">Research Paper Summarizer</span>
              </Link>
            </div>
          </div>
          <nav className="flex items-center space-x-4">
            <Link href="/" className="flex items-center text-gray-600 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
              <FiHome className="mr-1" />
              Home
            </Link>
            <Link href="/about" className="flex items-center text-gray-600 hover:text-primary-600 px-3 py-2 rounded-md text-sm font-medium">
              <FiInfo className="mr-1" />
              About
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header 
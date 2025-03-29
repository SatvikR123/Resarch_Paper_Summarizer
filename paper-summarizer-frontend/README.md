# Research Paper Summarizer

A modern web application for summarizing research papers using a fine-tuned BART model.

## Features

- Upload research papers in PDF format
- Get AI-generated summaries in seconds
- Extract images and figures from papers
- Copy and download summaries
- User-friendly modern interface

## Project Structure

```
paper-summarizer-frontend/
├── api/                   # Backend API
│   ├── server.py          # FastAPI server
│   └── requirements.txt   # Python dependencies
├── components/            # React components
├── pages/                 # Next.js pages
├── public/                # Static assets
└── styles/                # CSS styles
```

## Setup and Running

### Backend API

1. Install dependencies:

```bash
cd api
pip install -r requirements.txt
```

2. Start the API server:

```bash
python server.py
```

The API will be available at http://localhost:8000

### Frontend Application

1. Install dependencies:

```bash
npm install
```

2. Run the development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:3000

## Building for Production

```bash
npm run build
npm start
```

## Technology Stack

- **Frontend**: Next.js, React, TailwindCSS
- **Backend**: FastAPI
- **ML Model**: BART with LoRA fine-tuning
- **PDF Processing**: PyMuPDF 
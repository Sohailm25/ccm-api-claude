# Content API

A RESTful API service that allows users to store content, perform semantic searches, and get weekly summaries using Claude 3.7 Sonnet.

## Features

- Store content with URL, text, and thoughts
- Perform semantic searches using pgvector
- Generate intelligent weekly summaries using Claude 3.7 Sonnet
- Secure API key authentication
- Rate limiting protection

## Setup

### Local Development

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your credentials
3. Run with Docker Compose:
   ```
   docker-compose up
   ```
4. The API will be available at `http://localhost:8000`

### API Documentation

Once running, view the API documentation at `http://localhost:8000/docs`

## Deployment

This project is configured for deployment on Railway.app. 
# JIRA Ticket Analyzer

A Django web application for analyzing JIRA tickets using NLP techniques.

## Features
- User authentication
- JIRA data import (Excel/CSV)
- Natural Language Processing analysis
- Insights dashboard

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Setup the PostgreSQL database and update .env file
5. Run migrations:
   ```
   python manage.py migrate
   ```
6. Create a superuser:
   ```
   python manage.py createsuperuser
   ```
7. Run the server:
   ```
   python manage.py runserver
   ```

## Environment Variables (.env)
```
SECRET_KEY=your-secret-key
DEBUG=True
DB_NAME=jira_analyzer
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432
``` 

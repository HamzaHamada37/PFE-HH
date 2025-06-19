# RAG (Retrieval-Augmented Generation) Implementation Guide

## Overview

This implementation enhances your AI chatbot with RAG functionality, making it grounded in your actual business data from client analysis, team surveys, and business insights. The chatbot can now retrieve relevant information from your knowledge base before generating responses.

## 🚀 Features Implemented

### Core RAG Components
- **RAGService**: Core vector storage and retrieval using ChromaDB
- **Document Processors**: Convert business data into searchable documents
- **Query Processor**: Intelligent query classification and context retrieval
- **Vector Store Manager**: Enhanced ChromaDB management with filtering

### Data Sources Integrated
- **Client Analysis Data**: JIRA analysis results, client metrics, sentiment analysis
- **Survey Data**: Team satisfaction surveys, KPI metrics, feedback themes
- **Client Notes**: Qualitative notes and observations
- **Business Insights**: Actionable insights and recommendations

### Enhanced Chatbot Features
- **Context-Aware Responses**: Uses retrieved data to provide specific answers
- **Source Attribution**: Shows which data sources informed the response
- **Confidence Scoring**: Indicates reliability of retrieved information
- **Multi-Modal AI**: Works with both Ollama LLM and fallback AI service

## 📦 Dependencies

The following packages have been added to `requirements.txt`:
```
chromadb>=0.4.0
sentence-transformers>=2.2.2  # Already present
langchain>=0.1.0              # Already present
```

## 🛠️ Installation & Setup

### 1. Install Dependencies
```bash
pip install chromadb
```

### 2. Run Database Migrations
```bash
python manage.py migrate
```

### 3. Index Your Data
```bash
# Index data for all users
python manage.py index_data --verbose

# Index data for specific user
python manage.py index_data --user your_username --verbose

# Clear existing index and rebuild
python manage.py index_data --clear --verbose

# Show indexing statistics
python manage.py index_data --stats
```

### 4. Test the Implementation
```bash
python test_rag_implementation.py
```

## 🏗️ Architecture

### File Structure
```
analyzer/
├── rag_service.py              # Core RAG functionality
├── document_processors.py     # Data processing for indexing
├── vector_store.py            # Enhanced ChromaDB wrapper
├── chatbot_service.py         # Enhanced with RAG integration
├── management/commands/
│   └── index_data.py          # Data indexing command
└── models.py                  # Enhanced with RAG metadata
```

### Data Flow
1. **Data Processing**: Business data → Document processors → Structured documents
2. **Indexing**: Documents → Embeddings → ChromaDB vector store
3. **Query Processing**: User query → Query classification → Relevant document retrieval
4. **Response Generation**: Retrieved context + Query → LLM → Enhanced response

## 🔧 Configuration

### Vector Store Settings
- **Embedding Model**: `all-MiniLM-L6-v2` (lightweight, fast)
- **Vector Database**: ChromaDB (persistent, local storage)
- **Storage Location**: `{PROJECT_ROOT}/chroma_db/`

### Document Types
- `client_data`: Individual client metrics and analysis
- `analysis_summary`: Overall analysis summaries
- `business_insights`: Actionable insights and recommendations
- `survey_summary`: Team survey overviews
- `team_kpis`: KPI metrics and performance data
- `team_feedback`: Qualitative feedback themes
- `client_notes`: User-created client notes

## 🎯 Usage

### Web Interface
1. Navigate to the AI Agent page
2. Ask questions about your data:
   - "What are my worst performing clients?"
   - "Show me team satisfaction trends"
   - "What insights do you have about client sentiment?"
3. Responses will show RAG metadata indicating data sources used

### API Endpoints
```python
# Chat with RAG enhancement
POST /api/chat/message/
{
    "message": "Your question",
    "session_id": "optional_session_id"
}

# Refresh RAG index
POST /api/rag/refresh/

# Get RAG statistics
GET /api/rag/stats/

# Search knowledge base directly
POST /api/rag/search/
{
    "query": "search terms",
    "n_results": 5
}
```

### Management Commands
```bash
# Index all data
python manage.py index_data

# Index for specific user with verbose output
python manage.py index_data --user admin --verbose

# Clear and rebuild index
python manage.py index_data --clear

# Show statistics
python manage.py index_data --stats
```

## 🔍 How It Works

### Query Processing
1. **Classification**: Determines query type (client_analysis, team_analysis, etc.)
2. **Filtering**: Applies metadata filters based on query type
3. **Retrieval**: Searches vector store for relevant documents
4. **Context Building**: Combines retrieved documents into context
5. **Response Generation**: LLM generates response using retrieved context

### Document Processing
1. **Client Data**: Processes AnalysisResult objects into searchable documents
2. **Survey Data**: Converts SurveyAnalysis into KPI and feedback documents
3. **Notes**: Transforms ClientNote objects into searchable text
4. **Metadata**: Adds rich metadata for filtering and attribution

### Vector Storage
1. **Embeddings**: Uses sentence-transformers to create vector representations
2. **Storage**: ChromaDB stores vectors with metadata
3. **Retrieval**: Semantic search finds relevant documents
4. **Filtering**: Metadata filters improve relevance

## 🚨 Troubleshooting

### Common Issues

**RAG Service Not Available**
- Ensure ChromaDB is installed: `pip install chromadb`
- Check if sentence-transformers is working
- Verify ChromaDB directory permissions

**No Documents Found**
- Upload and process JIRA files or survey data first
- Run `python manage.py index_data --stats` to check data availability
- Ensure data processing completed successfully

**Low Quality Responses**
- Check RAG confidence scores in response metadata
- Verify relevant data exists for the query type
- Consider refreshing the index with `python manage.py index_data --clear`

**Performance Issues**
- ChromaDB creates indexes on first use (may be slow initially)
- Consider using smaller batch sizes for large datasets
- Monitor memory usage with large document collections

### Debug Commands
```bash
# Test RAG implementation
python test_rag_implementation.py

# Check data availability
python manage.py index_data --stats

# Verbose indexing with error details
python manage.py index_data --verbose

# Django shell for manual testing
python manage.py shell
>>> from analyzer.rag_service import RAGService
>>> from django.contrib.auth.models import User
>>> user = User.objects.first()
>>> rag = RAGService(user)
>>> rag.is_available()
```

## 🔮 Future Enhancements

### Planned Features
- **Real-time Indexing**: Auto-index new data as it's processed
- **Advanced Filtering**: Date ranges, client-specific searches
- **Hybrid Search**: Combine semantic and keyword search
- **Response Caching**: Cache frequent queries for better performance
- **Analytics Dashboard**: Track RAG usage and effectiveness

### Customization Options
- **Different Embedding Models**: Swap out sentence-transformers model
- **Custom Document Types**: Add new data source processors
- **Query Templates**: Pre-defined query patterns for common use cases
- **Response Formatting**: Custom response templates based on query type

## 📊 Monitoring

### Key Metrics
- **Index Size**: Number of documents in vector store
- **Query Performance**: Response time and relevance
- **Confidence Scores**: Quality of retrieved context
- **Usage Patterns**: Most common query types

### Health Checks
- RAG service availability
- Vector store connectivity
- Document processing status
- Embedding model performance

## 🤝 Contributing

When adding new data sources:
1. Create a new processor in `document_processors.py`
2. Add document type to metadata schema
3. Update query classification logic
4. Test with sample data
5. Update documentation

## 📝 License

This RAG implementation follows the same license as the main project.

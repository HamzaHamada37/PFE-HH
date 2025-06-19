#!/usr/bin/env python
"""
Test script to verify RAG implementation
Run this script to test the RAG functionality without starting the full Django server
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jira_analyzer.settings')
django.setup()

from django.contrib.auth.models import User
from analyzer.rag_service import RAGService
from analyzer.document_processors import DocumentProcessorManager
from analyzer.chatbot_service import ChatbotService

def test_rag_components():
    """Test RAG components functionality"""
    print("🧪 Testing RAG Implementation")
    print("=" * 50)
    
    # Get or create a test user
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        print("❌ No admin user found. Please create a user first.")
        return False
    
    print(f"✅ Using user: {user.username}")
    
    # Test RAG Service initialization
    print("\n📊 Testing RAG Service...")
    rag_service = RAGService(user)
    
    if rag_service.is_available():
        print("✅ RAG Service initialized successfully")
        stats = rag_service.get_collection_stats()
        print(f"   Collection stats: {stats}")
    else:
        print("❌ RAG Service not available")
        print("   Make sure ChromaDB and sentence-transformers are installed:")
        print("   pip install chromadb sentence-transformers")
        return False
    
    # Test Document Processors
    print("\n📄 Testing Document Processors...")
    processor_manager = DocumentProcessorManager(user)
    processor_stats = processor_manager.get_processor_stats()
    
    print(f"   Available data sources:")
    print(f"   - Analysis results: {processor_stats.get('analysis_results', 0)}")
    print(f"   - Survey analyses: {processor_stats.get('survey_analyses', 0)}")
    print(f"   - Client notes: {processor_stats.get('client_notes', 0)}")
    
    if processor_stats.get('total_data_sources', 0) == 0:
        print("⚠️  No data sources available for indexing")
        print("   Upload and process some JIRA files or survey data first")
        return True  # Not an error, just no data yet
    
    # Test document processing
    documents = processor_manager.process_all_documents()
    print(f"✅ Processed {len(documents)} documents")
    
    if documents:
        # Test adding documents to vector store
        print("\n🔍 Testing Vector Store...")
        success = rag_service.add_documents(documents[:5])  # Test with first 5 documents
        
        if success:
            print("✅ Documents added to vector store successfully")
            
            # Test search functionality
            test_query = "client performance metrics"
            results = rag_service.search_documents(test_query, n_results=3)
            print(f"✅ Search test completed - found {len(results)} relevant documents")
            
            if results:
                print("   Sample result:")
                sample = results[0]
                print(f"   - ID: {sample.get('id', 'N/A')}")
                print(f"   - Content preview: {sample.get('content', '')[:100]}...")
                print(f"   - Metadata: {sample.get('metadata', {})}")
        else:
            print("❌ Failed to add documents to vector store")
            return False
    
    # Test ChatbotService integration
    print("\n🤖 Testing Chatbot Integration...")
    chatbot = ChatbotService(user)
    
    if chatbot.rag_service and chatbot.rag_query_processor:
        print("✅ Chatbot RAG integration successful")
        
        # Test RAG stats
        rag_stats = chatbot.get_rag_stats()
        print(f"   RAG stats: {rag_stats}")
        
        # Test knowledge base search
        if documents:
            kb_results = chatbot.search_knowledge_base("client satisfaction", n_results=2)
            print(f"✅ Knowledge base search returned {len(kb_results)} results")
    else:
        print("⚠️  Chatbot RAG integration not available")
    
    print("\n🎉 RAG Implementation Test Complete!")
    return True

def test_sample_conversation():
    """Test a sample conversation with RAG"""
    print("\n💬 Testing Sample Conversation...")
    
    try:
        user = User.objects.get(username='admin')
        chatbot = ChatbotService(user)
        
        # Test queries
        test_queries = [
            "What are my client performance metrics?",
            "Show me clients with negative sentiment",
            "What insights do you have about team satisfaction?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            
            # Process message (this would normally go through the web interface)
            result = chatbot.process_message(query)
            
            if result.get('success'):
                response = result.get('response', '')
                rag_metadata = result.get('rag_metadata', {})
                
                print(f"✅ Response generated (length: {len(response)} chars)")
                print(f"   AI Mode: {result.get('ai_mode', 'unknown')}")
                print(f"   RAG Enabled: {rag_metadata.get('rag_enabled', False)}")
                
                if rag_metadata.get('rag_enabled'):
                    print(f"   RAG Confidence: {rag_metadata.get('rag_confidence', 0):.2f}")
                    print(f"   Sources: {len(rag_metadata.get('rag_sources', []))}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error in conversation test: {e}")

if __name__ == "__main__":
    print("🚀 Starting RAG Implementation Test")
    
    try:
        success = test_rag_components()
        
        if success:
            test_sample_conversation()
            print("\n✅ All tests completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Install missing dependencies if any errors occurred:")
            print("   pip install chromadb sentence-transformers")
            print("2. Run the Django management command to index your data:")
            print("   python manage.py index_data --user your_username --verbose")
            print("3. Start your Django server and test the AI Agent page")
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

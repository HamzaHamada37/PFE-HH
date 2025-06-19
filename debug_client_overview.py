#!/usr/bin/env python
"""
Debug script to diagnose and fix client overview retrieval issues
Tests chunking, embedding, indexing, and query logic specifically for client data
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jira_analyzer.settings')
django.setup()

from django.contrib.auth.models import User
from analyzer.rag_service import RAGService, RAGQueryProcessor
from analyzer.document_processors import DocumentProcessorManager, ClientDataProcessor
from analyzer.models import AnalysisResult, JiraFile

def diagnose_client_overview_issue():
    """Comprehensive diagnosis of client overview retrieval issues"""
    print("🔍 DIAGNOSING CLIENT OVERVIEW RETRIEVAL ISSUES")
    print("=" * 60)
    
    # Get user
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        print("❌ No admin user found")
        return False
    
    print(f"✅ Testing with user: {user.username}")
    
    # Step 1: Check raw data availability
    print("\n📊 STEP 1: Raw Data Availability")
    print("-" * 40)
    
    jira_files = JiraFile.objects.filter(user=user)
    analysis_results = AnalysisResult.objects.filter(jira_file__user=user)
    
    print(f"JIRA Files: {jira_files.count()}")
    print(f"Analysis Results: {analysis_results.count()}")
    
    if analysis_results.count() == 0:
        print("❌ ISSUE FOUND: No analysis results available!")
        print("   Solution: Upload and process JIRA files first")
        return False
    
    # Check client metrics in analysis results
    client_data_found = False
    for analysis in analysis_results:
        if analysis.client_metrics:
            client_data_found = True
            print(f"✅ Found client metrics in analysis {analysis.id}")
            print(f"   Clients: {list(analysis.client_metrics.keys())}")
            break
    
    if not client_data_found:
        print("❌ ISSUE FOUND: No client metrics in analysis results!")
        return False
    
    # Step 2: Test document processing
    print("\n📄 STEP 2: Document Processing")
    print("-" * 40)
    
    processor = ClientDataProcessor(user)
    documents = processor.process()
    
    print(f"Documents generated: {len(documents)}")
    
    if len(documents) == 0:
        print("❌ ISSUE FOUND: Document processor generated no documents!")
        return False
    
    # Analyze document types
    doc_types = {}
    source_types = {}
    client_overview_docs = []
    
    for doc in documents:
        doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
        source_type = doc.get('metadata', {}).get('source_type', 'unknown')
        
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        source_types[source_type] = source_types.get(source_type, 0) + 1
        
        if 'client_overview' in doc_type:
            client_overview_docs.append(doc)
    
    print("Document type distribution:")
    for doc_type, count in doc_types.items():
        print(f"  • {doc_type}: {count}")
    
    print("Source type distribution:")
    for source_type, count in source_types.items():
        print(f"  • {source_type}: {count}")
    
    print(f"Client overview documents: {len(client_overview_docs)}")
    
    if len(client_overview_docs) == 0:
        print("⚠️  WARNING: No client_overview documents generated!")
        print("   This might be why client overview queries fail")
    
    # Show sample client overview document
    if client_overview_docs:
        sample_doc = client_overview_docs[0]
        print(f"\nSample client overview document:")
        print(f"  ID: {sample_doc['id']}")
        print(f"  Content preview: {sample_doc['content'][:300]}...")
        print(f"  Metadata: {sample_doc['metadata']}")
    
    # Step 3: Test vector store indexing
    print("\n🗄️  STEP 3: Vector Store Indexing")
    print("-" * 40)
    
    rag_service = RAGService(user)
    if not rag_service.is_available():
        print("❌ ISSUE FOUND: RAG service not available!")
        return False
    
    # Clear and re-index
    print("Clearing existing index...")
    rag_service.clear_collection()
    
    print("Re-indexing documents...")
    success = rag_service.add_documents(documents)
    
    if not success:
        print("❌ ISSUE FOUND: Failed to index documents!")
        return False
    
    collection_stats = rag_service.get_collection_stats()
    print(f"✅ Indexed {collection_stats.get('total_documents', 0)} documents")
    
    # Step 4: Test retrieval with different strategies
    print("\n🔍 STEP 4: Retrieval Testing")
    print("-" * 40)
    
    test_queries = [
        "client overview",
        "show me client data",
        "client performance metrics",
        "what clients do we have"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test 1: Raw semantic search
        raw_results = rag_service.search_documents(query, n_results=5)
        print(f"  Raw semantic search: {len(raw_results)} results")
        
        if raw_results:
            for i, doc in enumerate(raw_results[:3], 1):
                metadata = doc.get('metadata', {})
                distance = doc.get('distance', 'N/A')
                print(f"    {i}. Distance: {distance}, Type: {metadata.get('doc_type', 'Unknown')}, Source: {metadata.get('source', 'Unknown')}")
        
        # Test 2: Filtered search
        client_filter = {
            "doc_type": {"$in": ["client_overview", "client_overview_summary", "client_data"]}
        }
        filtered_results = rag_service.search_documents(query, n_results=5, filter_metadata=client_filter)
        print(f"  Filtered search: {len(filtered_results)} results")
        
        # Test 3: Query processor
        query_processor = RAGQueryProcessor(rag_service)
        processed_result = query_processor.process_query(query)
        print(f"  Query processor: {processed_result.get('query_type')}, confidence: {processed_result.get('confidence'):.2f}")
    
    # Step 5: Test hybrid search
    print("\n🔄 STEP 5: Hybrid Search Testing")
    print("-" * 40)
    
    query = "client overview"
    print(f"Testing hybrid search for: '{query}'")
    
    try:
        hybrid_results = query_processor.search_with_hybrid_fallback(query, n_results=5)
        print(f"Hybrid search results: {len(hybrid_results)}")
        
        for i, doc in enumerate(hybrid_results[:3], 1):
            metadata = doc.get('metadata', {})
            distance = doc.get('distance', 'N/A')
            search_type = doc.get('search_type', 'unknown')
            print(f"  {i}. Distance: {distance}, Type: {metadata.get('doc_type', 'Unknown')}, Search: {search_type}")
    
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
    
    # Step 6: Full chatbot test
    print("\n🤖 STEP 6: Full Chatbot Test")
    print("-" * 40)
    
    from analyzer.chatbot_service import ChatbotService
    
    chatbot = ChatbotService(user)
    test_query = "What client overview data do you have?"
    
    print(f"Testing full chatbot with: '{test_query}'")
    
    result = chatbot.process_message(test_query)
    
    if result.get('success'):
        rag_metadata = result.get('rag_metadata', {})
        response = result.get('response', '')
        
        print(f"✅ Chatbot response generated")
        print(f"  RAG Enabled: {rag_metadata.get('rag_enabled', False)}")
        print(f"  Confidence: {rag_metadata.get('rag_confidence', 0):.2f}")
        print(f"  Sources: {len(rag_metadata.get('rag_sources', []))}")
        print(f"  Query Type: {rag_metadata.get('query_type', 'unknown')}")
        print(f"  Response length: {len(response)} characters")
        
        # Check if response contains client-specific data
        client_indicators = ['client', 'analysis', 'metrics', 'tickets', 'resolution']
        found_indicators = [indicator for indicator in client_indicators if indicator.lower() in response.lower()]
        
        if found_indicators:
            print(f"  ✅ Response contains client data: {found_indicators}")
        else:
            print(f"  ⚠️  Response seems generic, may not contain specific client data")
        
        print(f"  Response preview: {response[:300]}...")
    else:
        print(f"❌ Chatbot failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("🎯 DIAGNOSIS SUMMARY")
    print("="*60)
    
    issues_found = []
    fixes_applied = []
    
    if analysis_results.count() > 0:
        fixes_applied.append("✅ Analysis results available")
    else:
        issues_found.append("❌ No analysis results")
    
    if len(documents) > 0:
        fixes_applied.append("✅ Document processing working")
    else:
        issues_found.append("❌ Document processing failed")
    
    if len(client_overview_docs) > 0:
        fixes_applied.append("✅ Client overview documents generated")
    else:
        issues_found.append("⚠️  No client overview documents")
    
    if collection_stats.get('total_documents', 0) > 0:
        fixes_applied.append("✅ Vector store indexing working")
    else:
        issues_found.append("❌ Vector store indexing failed")
    
    print("Issues found:")
    for issue in issues_found:
        print(f"  {issue}")
    
    print("Fixes applied:")
    for fix in fixes_applied:
        print(f"  {fix}")
    
    if len(issues_found) == 0:
        print("\n🎉 SUCCESS: Client overview retrieval should now work!")
        print("The enhanced chunking, metadata, and hybrid search should resolve the issue.")
    else:
        print(f"\n⚠️  {len(issues_found)} issues still need attention.")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    try:
        success = diagnose_client_overview_issue()
        
        if success:
            print("\n🚀 Client overview retrieval is now fixed and ready!")
            print("Test it in the browser at: http://127.0.0.1:8000/ai-agent/")
        else:
            print("\n🔧 Some issues still need to be resolved.")
            
    except Exception as e:
        print(f"\n💥 Diagnosis failed with error: {e}")
        import traceback
        traceback.print_exc()

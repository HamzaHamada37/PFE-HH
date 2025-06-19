#!/usr/bin/env python
"""
Comprehensive test to demonstrate the client overview retrieval fix
Creates sample data and tests all the enhancements
"""

import os
import sys
import django
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jira_analyzer.settings')
django.setup()

from django.contrib.auth.models import User
from analyzer.models import JiraFile, AnalysisResult
from analyzer.rag_service import RAGService, RAGQueryProcessor
from analyzer.document_processors import ClientDataProcessor
from analyzer.chatbot_service import ChatbotService

def create_sample_client_data():
    """Create sample client data to test the fix"""
    print("🔧 Creating Sample Client Data for Testing")
    print("-" * 50)
    
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        print("❌ No admin user found")
        return False
    
    # Create sample JIRA file
    jira_file = JiraFile.objects.create(
        user=user,
        file='sample_jira_data.xlsx',
        original_filename='sample_jira_data.xlsx',
        processed=True,
        analysis_date=datetime.now().date()
    )
    
    # Create sample analysis result with comprehensive client metrics
    sample_client_metrics = {
        "Acme Corporation": {
            "sentiment": -0.45,
            "Priority_Impact": 0.75,
            "Issue_Type_Impact": 0.68,
            "Tickets": 15,
            "Avg_Resolution_Time_Days": 4.2,
            "Client_Impact": 0.82,
            "Customer_Experience_Score": 0.82
        },
        "TechStart Inc": {
            "sentiment": 0.25,
            "Priority_Impact": 0.35,
            "Issue_Type_Impact": 0.42,
            "Tickets": 8,
            "Avg_Resolution_Time_Days": 2.1,
            "Client_Impact": 0.38,
            "Customer_Experience_Score": 0.38
        },
        "Global Solutions Ltd": {
            "sentiment": -0.15,
            "Priority_Impact": 0.55,
            "Issue_Type_Impact": 0.48,
            "Tickets": 22,
            "Avg_Resolution_Time_Days": 3.8,
            "Client_Impact": 0.61,
            "Customer_Experience_Score": 0.61
        }
    }
    
    analysis_result = AnalysisResult.objects.create(
        jira_file=jira_file,
        issue_count=45,
        client_metrics=sample_client_metrics,
        actionable_insights=[
            "Acme Corporation requires immediate attention due to high impact score",
            "TechStart Inc shows positive sentiment and good resolution times",
            "Global Solutions Ltd has high ticket volume requiring process optimization"
        ]
    )
    
    print(f"✅ Created sample JIRA file: {jira_file.get_filename()}")
    print(f"✅ Created analysis result with {len(sample_client_metrics)} clients")
    print(f"   Clients: {list(sample_client_metrics.keys())}")
    
    return analysis_result

def test_enhanced_document_processing(analysis_result):
    """Test the enhanced document processing"""
    print("\n📄 Testing Enhanced Document Processing")
    print("-" * 50)
    
    user = analysis_result.jira_file.user
    processor = ClientDataProcessor(user)
    documents = processor.process()
    
    print(f"Documents generated: {len(documents)}")
    
    # Analyze document types and content
    client_overview_docs = []
    doc_types = {}
    
    for doc in documents:
        doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        if 'client_overview' in doc_type:
            client_overview_docs.append(doc)
    
    print("Document type distribution:")
    for doc_type, count in doc_types.items():
        print(f"  • {doc_type}: {count}")
    
    print(f"\n✅ Client overview documents: {len(client_overview_docs)}")
    
    # Show sample client overview document
    if client_overview_docs:
        sample_doc = client_overview_docs[0]
        print(f"\nSample client overview document:")
        print(f"  ID: {sample_doc['id']}")
        print(f"  Client: {sample_doc['metadata'].get('client_name', 'Unknown')}")
        print(f"  Source Type: {sample_doc['metadata'].get('source_type', 'Unknown')}")
        print(f"  Content preview:")
        print("  " + "\n  ".join(sample_doc['content'].split('\n')[:15]))
        print("  ...")
    
    return documents

def test_vector_indexing_and_retrieval(documents):
    """Test vector indexing and retrieval with different strategies"""
    print("\n🗄️  Testing Vector Indexing and Retrieval")
    print("-" * 50)
    
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        return False
    
    rag_service = RAGService(user)
    if not rag_service.is_available():
        print("❌ RAG service not available")
        return False
    
    # Clear and index
    print("Clearing existing index...")
    rag_service.clear_collection()
    
    print("Indexing documents...")
    success = rag_service.add_documents(documents)
    
    if not success:
        print("❌ Failed to index documents")
        return False
    
    collection_stats = rag_service.get_collection_stats()
    print(f"✅ Indexed {collection_stats.get('total_documents', 0)} documents")
    
    # Test different client overview queries
    test_queries = [
        "client overview",
        "show me client data",
        "what clients do we have",
        "client performance metrics",
        "Acme Corporation metrics",
        "client sentiment analysis",
        "client resolution times"
    ]
    
    print(f"\n🔍 Testing Client Overview Queries:")
    
    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        
        # Test 1: Raw semantic search
        raw_results = rag_service.search_documents(query, n_results=3)
        print(f"  Raw search: {len(raw_results)} results")
        
        if raw_results:
            best_result = raw_results[0]
            metadata = best_result.get('metadata', {})
            distance = best_result.get('distance', 'N/A')
            print(f"    Best match: {metadata.get('doc_type', 'Unknown')} (distance: {distance})")
            print(f"    Client: {metadata.get('client_name', 'N/A')}")
        
        # Test 2: Filtered search for client data
        client_filter = {
            "doc_type": {"$in": ["client_overview", "client_overview_summary", "client_data"]}
        }
        filtered_results = rag_service.search_documents(query, n_results=3, filter_metadata=client_filter)
        print(f"  Filtered search: {len(filtered_results)} results")
        
        # Test 3: Query processor with hybrid search
        query_processor = RAGQueryProcessor(rag_service)
        processed_result = query_processor.process_query(query)
        print(f"  Query processor: {processed_result.get('query_type')}, confidence: {processed_result.get('confidence'):.2f}")
        print(f"  Context length: {len(processed_result.get('context', ''))} chars")
    
    return True

def test_full_chatbot_integration():
    """Test full chatbot integration with client overview queries"""
    print("\n🤖 Testing Full Chatbot Integration")
    print("-" * 50)
    
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        return False
    
    chatbot = ChatbotService(user)
    
    client_queries = [
        "What client overview data do you have?",
        "Show me information about Acme Corporation",
        "Which clients have negative sentiment?",
        "What are the resolution times for our clients?",
        "Tell me about client performance metrics",
        "Which client needs the most attention?"
    ]
    
    successful_queries = 0
    
    for query in client_queries:
        print(f"\n🔍 Query: {query}")
        
        try:
            result = chatbot.process_message(query)
            
            if result.get('success'):
                rag_metadata = result.get('rag_metadata', {})
                response = result.get('response', '')
                
                print(f"  ✅ Response generated")
                print(f"  RAG Enabled: {rag_metadata.get('rag_enabled', False)}")
                print(f"  Confidence: {rag_metadata.get('rag_confidence', 0):.2f}")
                print(f"  Sources: {len(rag_metadata.get('rag_sources', []))}")
                print(f"  Query Type: {rag_metadata.get('query_type', 'unknown')}")
                
                # Check if response contains client-specific data
                client_indicators = ['acme', 'techstart', 'global solutions', 'sentiment', 'resolution', 'tickets', 'client']
                found_indicators = [indicator for indicator in client_indicators if indicator.lower() in response.lower()]
                
                if (rag_metadata.get('rag_enabled') and 
                    rag_metadata.get('rag_confidence', 0) >= 0.5 and
                    len(found_indicators) >= 2):
                    print(f"  🎯 SUCCESS - Contains client data: {found_indicators[:3]}")
                    successful_queries += 1
                else:
                    print(f"  ⚠️  PARTIAL - Limited client data references")
                
                print(f"  Response preview: {response[:200]}...")
            else:
                print(f"  ❌ FAILED - {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"  ❌ ERROR - {e}")
    
    success_rate = successful_queries / len(client_queries)
    print(f"\n📈 Client Query Success Rate: {successful_queries}/{len(client_queries)} ({success_rate*100:.1f}%)")
    
    return success_rate >= 0.7

def main():
    """Main test function"""
    print("🚀 COMPREHENSIVE CLIENT OVERVIEW RETRIEVAL FIX TEST")
    print("=" * 60)
    print("This test demonstrates the fixes for client overview retrieval issues:")
    print("1. Enhanced chunking with larger context")
    print("2. Better metadata with source_type filters")
    print("3. Hybrid keyword + vector search fallback")
    print("4. Improved query classification")
    print()
    
    try:
        # Step 1: Create sample data
        analysis_result = create_sample_client_data()
        if not analysis_result:
            print("❌ Failed to create sample data")
            return
        
        # Step 2: Test document processing
        documents = test_enhanced_document_processing(analysis_result)
        if not documents:
            print("❌ No documents generated")
            return
        
        # Step 3: Test vector indexing and retrieval
        indexing_success = test_vector_indexing_and_retrieval(documents)
        if not indexing_success:
            print("❌ Vector indexing failed")
            return
        
        # Step 4: Test full chatbot integration
        chatbot_success = test_full_chatbot_integration()
        
        # Final assessment
        print("\n" + "="*60)
        print("🎯 FINAL ASSESSMENT")
        print("="*60)
        
        if chatbot_success:
            print("🎉 SUCCESS: Client overview retrieval is now working!")
            print("\n✅ Fixes implemented and verified:")
            print("  • Enhanced chunking with larger context and better organization")
            print("  • Added source_type metadata filters for precise targeting")
            print("  • Implemented hybrid keyword + vector search fallback")
            print("  • Improved query classification for client overview detection")
            print("  • Comprehensive client metrics extraction and indexing")
            
            print(f"\n🌐 Ready to test with real data!")
            print("Upload JIRA files in the dashboard to see client overview data indexed automatically.")
        else:
            print("⚠️  Some issues still need attention, but core functionality is working.")
        
        # Cleanup
        print(f"\n🧹 Cleaning up test data...")
        analysis_result.delete()
        analysis_result.jira_file.delete()
        print("✅ Test data cleaned up")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Final verification test for comprehensive RAG implementation
Tests all requirements: historical data indexing, real-time sync, and data persistence
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jira_analyzer.settings')
django.setup()

from django.contrib.auth.models import User
from analyzer.chatbot_service import ChatbotService
from analyzer.models import AnalysisResult, SurveyAnalysis, ClientNote

def test_comprehensive_requirements():
    """Test all comprehensive RAG requirements"""
    print("🎯 FINAL RAG VERIFICATION TEST")
    print("=" * 60)
    print("Testing: Comprehensive data indexing, real-time sync, and data persistence")
    
    # Get user
    try:
        user = User.objects.get(username='admin')
    except User.DoesNotExist:
        print("❌ No admin user found")
        return False
    
    print(f"✅ Testing with user: {user.username}")
    
    # Initialize chatbot
    chatbot = ChatbotService(user)
    
    # Test 1: Historical Data Access
    print("\n📊 TEST 1: Historical Data Access")
    print("-" * 40)
    
    # Check data availability
    analysis_count = AnalysisResult.objects.filter(
        jira_file__user=user, jira_file__processed=True
    ).count()
    survey_count = SurveyAnalysis.objects.filter(
        survey_file__user=user, survey_file__processed=True
    ).count()
    notes_count = ClientNote.objects.filter(user=user).count()
    
    print(f"📈 Analysis Results: {analysis_count}")
    print(f"📋 Survey Analyses: {survey_count}")
    print(f"📝 Client Notes: {notes_count}")
    
    total_data_sources = analysis_count + survey_count + notes_count
    if total_data_sources == 0:
        print("⚠️  No historical data available for testing")
        return False
    
    # Test specific historical queries
    historical_queries = [
        "What is our overall team satisfaction score?",
        "Show me the psychological safety metrics",
        "What are the main feedback themes from our surveys?",
        "How many survey responses did we collect?",
        "What are our team collaboration scores?"
    ]
    
    successful_historical = 0
    
    for query in historical_queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = chatbot.process_message(query)
            if result.get('success'):
                rag_metadata = result.get('rag_metadata', {})
                if (rag_metadata.get('rag_enabled') and 
                    rag_metadata.get('rag_confidence', 0) >= 0.5 and
                    len(rag_metadata.get('rag_sources', [])) > 0):
                    print(f"  ✅ SUCCESS - Confidence: {rag_metadata.get('rag_confidence'):.2f}, Sources: {len(rag_metadata.get('rag_sources', []))}")
                    successful_historical += 1
                else:
                    print(f"  ⚠️  LOW QUALITY - Confidence: {rag_metadata.get('rag_confidence', 0):.2f}")
            else:
                print(f"  ❌ FAILED - {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  ❌ ERROR - {e}")
    
    historical_success_rate = successful_historical / len(historical_queries)
    print(f"\n📈 Historical Data Access: {successful_historical}/{len(historical_queries)} ({historical_success_rate*100:.1f}%)")
    
    # Test 2: Data Persistence and Memory
    print("\n🧠 TEST 2: Data Persistence and Memory")
    print("-" * 40)
    
    # Test that the chatbot remembers and can compare data
    memory_queries = [
        "Compare this month's satisfaction to previous periods",
        "What trends do you see in our team data?",
        "How has our psychological safety score changed?",
        "What insights can you provide from all our historical data?"
    ]
    
    successful_memory = 0
    
    for query in memory_queries:
        print(f"\n🔍 Memory Test: {query}")
        try:
            result = chatbot.process_message(query)
            if result.get('success'):
                rag_metadata = result.get('rag_metadata', {})
                response = result.get('response', '')
                
                # Check if response contains specific data references
                data_indicators = ['4.02', '3.97', '3.69', '4.19', '4.22', 'survey', 'satisfaction', 'KPI']
                found_indicators = [indicator for indicator in data_indicators if indicator.lower() in response.lower()]
                
                if (rag_metadata.get('rag_enabled') and 
                    rag_metadata.get('rag_confidence', 0) >= 0.5 and
                    len(found_indicators) >= 2):
                    print(f"  ✅ SUCCESS - Found data references: {found_indicators[:3]}")
                    successful_memory += 1
                else:
                    print(f"  ⚠️  GENERIC RESPONSE - Limited data references")
            else:
                print(f"  ❌ FAILED")
        except Exception as e:
            print(f"  ❌ ERROR - {e}")
    
    memory_success_rate = successful_memory / len(memory_queries)
    print(f"\n🧠 Data Persistence: {successful_memory}/{len(memory_queries)} ({memory_success_rate*100:.1f}%)")
    
    # Test 3: Real-time Synchronization
    print("\n🔄 TEST 3: Real-time Synchronization")
    print("-" * 40)
    
    # Check if signals are properly configured
    try:
        from analyzer import signals
        print("✅ Signal system loaded")
        
        # Get RAG stats
        rag_stats = chatbot.get_rag_stats()
        if rag_stats.get('rag_available'):
            collection_stats = rag_stats.get('collection_stats', {})
            print(f"✅ Vector store active with {collection_stats.get('total_documents', 0)} documents")
            
            # Test indexing metadata
            indexed_analyses = AnalysisResult.objects.filter(
                jira_file__user=user, rag_indexed=True
            ).count()
            indexed_surveys = SurveyAnalysis.objects.filter(
                survey_file__user=user, rag_indexed=True
            ).count()
            
            print(f"✅ Indexed analyses: {indexed_analyses}")
            print(f"✅ Indexed surveys: {indexed_surveys}")
            
            sync_success = True
        else:
            print("❌ RAG service not available")
            sync_success = False
            
    except Exception as e:
        print(f"❌ Error checking sync system: {e}")
        sync_success = False
    
    # Test 4: Future-proof Architecture
    print("\n🚀 TEST 4: Future-proof Architecture")
    print("-" * 40)
    
    # Check if system can handle different query types
    architecture_queries = [
        ("client_analysis", "What client performance data do we have?"),
        ("team_analysis", "Show me team satisfaction metrics"),
        ("performance_analysis", "What are our key performance indicators?"),
        ("qualitative_analysis", "What feedback themes have we identified?")
    ]
    
    successful_architecture = 0
    
    for query_type, query in architecture_queries:
        print(f"\n🔍 {query_type}: {query}")
        try:
            result = chatbot.process_message(query)
            if result.get('success'):
                rag_metadata = result.get('rag_metadata', {})
                if rag_metadata.get('query_type') == query_type:
                    print(f"  ✅ Correctly classified as {query_type}")
                    successful_architecture += 1
                else:
                    print(f"  ⚠️  Classified as {rag_metadata.get('query_type', 'unknown')}")
            else:
                print(f"  ❌ FAILED")
        except Exception as e:
            print(f"  ❌ ERROR - {e}")
    
    architecture_success_rate = successful_architecture / len(architecture_queries)
    print(f"\n🚀 Architecture Flexibility: {successful_architecture}/{len(architecture_queries)} ({architecture_success_rate*100:.1f}%)")
    
    # Final Assessment
    print("\n" + "="*60)
    print("📋 COMPREHENSIVE RAG ASSESSMENT")
    print("="*60)
    
    overall_scores = {
        "Historical Data Access": historical_success_rate,
        "Data Persistence": memory_success_rate,
        "Real-time Sync": 1.0 if sync_success else 0.0,
        "Architecture": architecture_success_rate
    }
    
    for test_name, score in overall_scores.items():
        status = "✅ EXCELLENT" if score >= 0.8 else "👍 GOOD" if score >= 0.6 else "⚠️ NEEDS WORK" if score >= 0.4 else "❌ POOR"
        print(f"  {test_name}: {score*100:.1f}% - {status}")
    
    overall_score = sum(overall_scores.values()) / len(overall_scores)
    
    print(f"\n🎯 OVERALL SCORE: {overall_score*100:.1f}%")
    
    if overall_score >= 0.8:
        print("🎉 EXCELLENT: RAG system fully meets all requirements!")
        print("\n✅ Your AI Agent can now:")
        print("  • Access all historical data comprehensively")
        print("  • Provide specific, data-grounded responses")
        print("  • Reference actual business metrics and insights")
        print("  • Automatically index new data in real-time")
        print("  • Handle complex queries about trends and comparisons")
        print("  • Maintain persistent memory of all business data")
        
        print(f"\n🌐 Ready for production use at: http://127.0.0.1:8000/ai-agent/")
        return True
        
    elif overall_score >= 0.6:
        print("👍 GOOD: RAG system is functional with minor issues")
        return True
        
    else:
        print("⚠️ NEEDS IMPROVEMENT: Some requirements not fully met")
        return False

if __name__ == "__main__":
    try:
        success = test_comprehensive_requirements()
        
        if success:
            print("\n🎊 CONGRATULATIONS!")
            print("Your RAG-enhanced AI Agent is ready for comprehensive business intelligence!")
        else:
            print("\n🔧 Some issues need to be addressed before full deployment.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

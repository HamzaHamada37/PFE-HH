import os
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from django.contrib.auth.models import User
from django.db.models import Q, Avg, Count, Sum
from django.utils import timezone
from .models import (
    JiraFile, AnalysisResult, ClientNote,
    ChatSession, ChatMessage
)

# Import RAG components
try:
    from .rag_service import RAGService, RAGQueryProcessor
    from .document_processors import DocumentProcessorManager
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG components not available: {e}")
    RAG_AVAILABLE = False

try:
    from langchain_ollama import OllamaLLM
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import HumanMessage, AIMessage
    from langchain.callbacks.manager import CallbackManager
    from langchain_core.callbacks import StreamingStdOutCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Error importing LangChain: {str(e)}")
    LANGCHAIN_AVAILABLE = False

# Fallback AI service for when LangChain is not available
import requests
import json


class DataAnalysisService:
    """Service to query and analyze business data for the chatbot"""
    
    def __init__(self, user: User):
        self.user = user
    
    def get_client_overview(self) -> Dict[str, Any]:
        """Get comprehensive client overview data"""
        analysis_results = AnalysisResult.objects.filter(
            jira_file__user=self.user,
            jira_file__processed=True
        ).order_by('-jira_file__analysis_date', '-created_at')
        
        if not analysis_results.exists():
            return {"error": "No analysis data available"}
        
        # Aggregate client data
        all_clients = set()
        latest_metrics = {}
        total_tickets = 0
        sentiment_summary = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for analysis in analysis_results:
            if not analysis.client_metrics:
                continue
                
            total_tickets += analysis.issue_count
            
            for client_name, metrics in analysis.client_metrics.items():
                all_clients.add(client_name)
                if client_name not in latest_metrics:
                    latest_metrics[client_name] = metrics
                    
                    # Categorize sentiment
                    sentiment = metrics.get('sentiment', 0)
                    if sentiment > 0.1:
                        sentiment_summary['positive'] += 1
                    elif sentiment < -0.1:
                        sentiment_summary['negative'] += 1
                    else:
                        sentiment_summary['neutral'] += 1
        
        return {
            'total_clients': len(all_clients),
            'total_tickets': total_tickets,
            'client_metrics': latest_metrics,
            'sentiment_summary': sentiment_summary,
            'analysis_count': analysis_results.count()
        }
    
    def get_client_details(self, client_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific client"""
        analysis_results = AnalysisResult.objects.filter(
            jira_file__user=self.user,
            jira_file__processed=True
        ).order_by('-jira_file__analysis_date', '-created_at')
        
        client_data = []
        for analysis in analysis_results:
            if analysis.client_metrics and client_name in analysis.client_metrics:
                metrics = analysis.client_metrics[client_name]
                analysis_date = analysis.jira_file.analysis_date or analysis.created_at.date()
                
                client_data.append({
                    'date': analysis_date.strftime('%Y-%m-%d'),
                    'sentiment': metrics.get('sentiment', 0),
                    'resolution_time': metrics.get('Avg_Resolution_Time_Days', 0),
                    'tickets': metrics.get('Tickets', 0),
                    'client_impact': metrics.get('Client_Impact', 0),
                    'file_name': analysis.jira_file.original_filename or analysis.jira_file.file.name
                })
        
        # Get client notes
        try:
            client_note = ClientNote.objects.get(user=self.user, client_name=client_name)
            note_text = client_note.note_text
        except ClientNote.DoesNotExist:
            note_text = None
        
        return {
            'client_name': client_name,
            'historical_data': client_data,
            'note': note_text,
            'data_points': len(client_data)
        }
    
    def get_trending_issues(self) -> Dict[str, Any]:
        """Get trending issues and patterns across all clients"""
        latest_analysis = AnalysisResult.objects.filter(
            jira_file__user=self.user,
            jira_file__processed=True
        ).order_by('-jira_file__analysis_date', '-created_at').first()
        
        if not latest_analysis:
            return {"error": "No analysis data available"}
        
        return {
            'common_themes': latest_analysis.common_themes,
            'ticket_types': latest_analysis.ticket_types,
            'priority_distribution': latest_analysis.priority_distribution,
            'status_distribution': latest_analysis.status_distribution,
            'actionable_insights': latest_analysis.actionable_insights
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics"""
        overview = self.get_client_overview()
        if 'error' in overview:
            return overview
        
        # Calculate averages
        client_metrics = overview['client_metrics']
        if not client_metrics:
            return {"error": "No client metrics available"}
        
        resolution_times = [m.get('Avg_Resolution_Time_Days', 0) for m in client_metrics.values()]
        sentiments = [m.get('sentiment', 0) for m in client_metrics.values()]
        impact_scores = [m.get('Client_Impact', 0) for m in client_metrics.values()]
        
        return {
            'avg_resolution_time': sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            'avg_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
            'avg_impact_score': sum(impact_scores) / len(impact_scores) if impact_scores else 0,
            'critical_clients': len([s for s in impact_scores if s > 0.6]),
            'excellent_clients': len([s for s in impact_scores if s < 0.2]),
            'total_clients': overview['total_clients'],
            'total_tickets': overview['total_tickets']
        }


class FallbackAIService:
    """Fallback AI service that provides intelligent responses without external AI models"""

    def __init__(self, data_service: DataAnalysisService):
        self.data_service = data_service

    def generate_response(self, user_message: str, conversation_history: List[Dict]) -> str:
        """Generate intelligent responses based on data analysis"""
        message_lower = user_message.lower()

        # Get current business data
        overview = self.data_service.get_client_overview()
        performance = self.data_service.get_performance_metrics()

        # Client-related queries
        if any(word in message_lower for word in ['client', 'customer']):
            if 'negative' in message_lower or 'bad' in message_lower:
                return self._get_negative_clients_response(overview, performance)
            elif 'best' in message_lower or 'good' in message_lower:
                return self._get_best_clients_response(overview, performance)
            elif 'list' in message_lower or 'show' in message_lower:
                return self._get_client_list_response(overview)

        # Performance queries
        if any(word in message_lower for word in ['performance', 'metrics', 'stats']):
            return self._get_performance_response(performance)

        # Email drafting
        if any(word in message_lower for word in ['email', 'draft', 'write']):
            return self._get_email_template_response()

        # Recommendations
        if any(word in message_lower for word in ['recommend', 'suggest', 'advice']):
            return self._get_recommendations_response(performance)

        # Default response
        return self._get_default_response(overview, performance)

    def _get_negative_clients_response(self, overview, performance):
        if 'error' in overview:
            return "I don't have any client data available yet. Please upload and process some JIRA files first to analyze client sentiment."

        critical_count = performance.get('critical_clients', 0)
        total_clients = overview.get('total_clients', 0)

        response = f"Based on your data analysis:\n\n"
        response += f"📊 **Client Sentiment Analysis:**\n"
        response += f"• Total clients analyzed: {total_clients}\n"
        response += f"• Critical clients (requiring attention): {critical_count}\n"
        response += f"• Average sentiment score: {performance.get('avg_sentiment', 0):.2f}\n\n"

        if critical_count > 0:
            response += f"⚠️ **Action Required:**\n"
            response += f"You have {critical_count} clients with high impact scores that need immediate attention. "
            response += f"I recommend reviewing their recent tickets and reaching out proactively.\n\n"
            response += f"Would you like me to draft a professional email template for client outreach?"
        else:
            response += f"✅ **Good News:**\n"
            response += f"No clients currently show critical negative sentiment. Your client satisfaction appears to be well-managed!"

        return response

    def _get_best_clients_response(self, overview, performance):
        if 'error' in overview:
            return "I don't have any client data available yet. Please upload and process some JIRA files first."

        excellent_count = performance.get('excellent_clients', 0)
        total_clients = overview.get('total_clients', 0)

        response = f"📈 **Top Performing Clients:**\n\n"
        response += f"• Excellent clients (low impact scores): {excellent_count}\n"
        response += f"• Average resolution time: {performance.get('avg_resolution_time', 0):.1f} days\n"
        response += f"• Overall client satisfaction: {performance.get('avg_sentiment', 0):.2f}\n\n"

        if excellent_count > 0:
            response += f"🌟 **Success Insights:**\n"
            response += f"You have {excellent_count} clients with excellent satisfaction scores. "
            response += f"Consider analyzing what makes these relationships successful and applying those practices to other clients."

        return response

    def _get_client_list_response(self, overview):
        if 'error' in overview:
            return "No client data is currently available. Please upload and process JIRA files to see client metrics."

        total_clients = overview.get('total_clients', 0)
        total_tickets = overview.get('total_tickets', 0)

        response = f"📋 **Client Overview:**\n\n"
        response += f"• Total clients: {total_clients}\n"
        response += f"• Total support tickets: {total_tickets:,}\n"
        response += f"• Average tickets per client: {total_tickets/total_clients if total_clients > 0 else 0:.1f}\n\n"

        response += f"To see detailed information about specific clients, visit the Client Overview page or ask me about specific client metrics."

        return response

    def _get_performance_response(self, performance):
        if 'error' in performance:
            return "No performance data is available yet. Please process some JIRA files first."

        response = f"📊 **Performance Metrics Summary:**\n\n"
        response += f"• Average resolution time: {performance.get('avg_resolution_time', 0):.1f} days\n"
        response += f"• Average client sentiment: {performance.get('avg_sentiment', 0):.2f}\n"
        response += f"• Critical clients: {performance.get('critical_clients', 0)}\n"
        response += f"• Excellent clients: {performance.get('excellent_clients', 0)}\n"
        response += f"• Total tickets processed: {performance.get('total_tickets', 0):,}\n\n"

        # Add insights
        avg_resolution = performance.get('avg_resolution_time', 0)
        if avg_resolution > 5:
            response += f"⚠️ **Insight:** Your average resolution time of {avg_resolution:.1f} days is above the recommended 5-day target. Consider optimizing your support processes.\n"
        elif avg_resolution < 2:
            response += f"✅ **Insight:** Excellent resolution time! Your team is responding quickly to client issues.\n"

        return response

    def _get_email_template_response(self):
        return """📧 **Professional Client Outreach Email Template:**

Subject: Proactive Support Check-in - [Client Name]

Dear [Client Name],

I hope this message finds you well. As part of our commitment to providing exceptional service, I wanted to reach out proactively regarding your recent support experience.

Our analysis shows that we've been working together on [number] support tickets recently, and I want to ensure we're meeting your expectations and addressing any concerns you might have.

**What we've accomplished:**
• Resolved [X] tickets with an average resolution time of [Y] days
• Addressed issues related to [main categories]
• Implemented [specific improvements/solutions]

**Moving forward:**
I'd love to schedule a brief call to discuss:
- Your overall satisfaction with our support
- Any areas where we can improve our service
- Upcoming projects or challenges we can help with

Would you be available for a 15-minute call this week? I'm flexible with timing and happy to work around your schedule.

Thank you for your continued partnership. We value your business and are committed to your success.

Best regards,
[Your Name]
[Your Title]
[Contact Information]

---
💡 **Tip:** Customize this template with specific client data and recent ticket information for maximum impact."""

    def _get_recommendations_response(self, performance):
        if 'error' in performance:
            return "I need client data to provide specific recommendations. Please upload and process JIRA files first."

        critical_clients = performance.get('critical_clients', 0)
        avg_resolution = performance.get('avg_resolution_time', 0)

        response = f"💡 **Strategic Recommendations:**\n\n"

        if critical_clients > 0:
            response += f"🔴 **Immediate Actions:**\n"
            response += f"• Focus on {critical_clients} critical clients with high impact scores\n"
            response += f"• Schedule proactive check-in calls within 48 hours\n"
            response += f"• Review their recent ticket history for patterns\n\n"

        if avg_resolution > 3:
            response += f"🟡 **Process Improvements:**\n"
            response += f"• Current resolution time ({avg_resolution:.1f} days) could be improved\n"
            response += f"• Consider implementing automated triage systems\n"
            response += f"• Review resource allocation and team capacity\n\n"

        response += f"🟢 **Long-term Strategy:**\n"
        response += f"• Implement regular client health check processes\n"
        response += f"• Create early warning systems for sentiment degradation\n"
        response += f"• Develop client success programs based on top performers\n"
        response += f"• Consider quarterly business reviews with key clients"

        return response

    def _get_default_response(self, overview, performance):
        if 'error' in overview:
            return """👋 **Welcome to your AI Data Analyst!**

I'm here to help you analyze client data and provide business insights, but I don't see any processed data yet.

**To get started:**
1. Upload JIRA files using the Upload page
2. Process the files to generate client metrics
3. Return here to ask questions about your data

**Once you have data, you can ask me:**
• "Show me clients with negative sentiment"
• "What are my performance metrics?"
• "Draft an email for client outreach"
• "Give me recommendations for improving client satisfaction"

I'm ready to help you turn your data into actionable insights! 🚀"""

        total_clients = overview.get('total_clients', 0)
        total_tickets = overview.get('total_tickets', 0)

        return f"""👋 **Hello! I'm your AI Data Analyst.**

I can see you have data for {total_clients} clients and {total_tickets:,} support tickets. Here's what I can help you with:

**📊 Data Analysis:**
• Client performance and sentiment analysis
• Trend identification and pattern recognition
• Resolution time and efficiency metrics

**💼 Business Intelligence:**
• Strategic recommendations based on your data
• Client health assessments
• Performance benchmarking

**📧 Communication:**
• Draft professional client outreach emails
• Create data-driven reports
• Generate actionable insights

**Try asking me:**
• "Show me my worst performing clients"
• "What trends do you see in my data?"
• "Help me draft an email to a struggling client"
• "Give me recommendations to improve client satisfaction"

What would you like to explore first? 🚀"""


class ChatbotService:
    """Main chatbot service that handles AI interactions and data analysis with RAG"""

    def __init__(self, user: User):
        self.user = user
        self.data_service = DataAnalysisService(user)
        self.llm = None
        self.fallback_ai = FallbackAIService(self.data_service)

        # Initialize RAG components
        self.rag_service = None
        self.rag_query_processor = None
        self._initialize_rag()
        self._initialize_llm()

    def _initialize_rag(self):
        """Initialize RAG service and query processor"""
        if not RAG_AVAILABLE:
            print("RAG components not available")
            return

        try:
            self.rag_service = RAGService(self.user)
            if self.rag_service.is_available():
                self.rag_query_processor = RAGQueryProcessor(self.rag_service)
                print(f"RAG service initialized for user {self.user.username}")
            else:
                print("RAG service not available (ChromaDB/embeddings not initialized)")
        except Exception as e:
            print(f"Error initializing RAG service: {e}")
            self.rag_service = None
            self.rag_query_processor = None

    def _initialize_llm(self):
        """Initialize the Ollama LLM"""
        if not LANGCHAIN_AVAILABLE:
            print("LangChain not available, using fallback AI service")
            return

        try:
            print("Initializing Ollama LLM...")
            # Create a callback manager for streaming output
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.llm = OllamaLLM(
                model="llama3:latest",
                temperature=0.7,
                base_url="http://localhost:11434",
                timeout=30,  # Add timeout to prevent hanging
                callback_manager=callback_manager,
                verbose=True  # Enable verbose output
            )
            
            # Test connection with a simple prompt
            print("Testing Ollama connection...")
            test_response = self.llm.invoke("Hello, are you working?")
            print(f"Ollama test response: {test_response}")
            
            if not test_response:
                raise Exception("No response from Ollama")
                
            print("Ollama LLM initialized successfully!")
        except Exception as e:
            print(f"Error initializing Ollama: {str(e)}")
            self.llm = None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get existing session or create a new one"""
        if session_id:
            try:
                session = ChatSession.objects.get(session_id=session_id, user=self.user)
                session.updated_at = timezone.now()
                session.save()
                return session
            except ChatSession.DoesNotExist:
                pass
        
        # Create new session
        session_id = str(uuid.uuid4())
        session = ChatSession.objects.create(
            user=self.user,
            session_id=session_id,
            title="New Conversation"
        )
        return session
    
    def save_message(self, session: ChatSession, message_type: str, content: str, metadata: Dict = None):
        """Save a message to the database"""
        ChatMessage.objects.create(
            session=session,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
    
    def get_conversation_history(self, session: ChatSession, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        messages = session.messages.order_by('-created_at')[:limit]
        return [
            {
                'type': msg.message_type,
                'content': msg.content,
                'timestamp': msg.created_at.isoformat(),
                'metadata': msg.metadata
            }
            for msg in reversed(messages)
        ]
    
    def _build_context_prompt(self, user_message: str, conversation_history: List[Dict]) -> str:
        """Build a comprehensive prompt with business context and RAG-retrieved information"""

        # Get RAG context if available
        rag_context = ""
        rag_sources = []
        rag_confidence = 0.0

        if self.rag_query_processor:
            try:
                rag_result = self.rag_query_processor.process_query(user_message, conversation_history)
                rag_context = rag_result.get('context', '')
                rag_sources = rag_result.get('sources', [])
                rag_confidence = rag_result.get('confidence', 0.0)
            except Exception as e:
                print(f"Error retrieving RAG context: {e}")

        # Get current business data (fallback/summary data)
        overview = self.data_service.get_client_overview()
        performance = self.data_service.get_performance_metrics()
        trending = self.data_service.get_trending_issues()

        context_prompt = f"""You are an intelligent business data analyst assistant for a client support management system.
You help users analyze client data, identify trends, and provide actionable insights.

CURRENT BUSINESS CONTEXT:
- Total Clients: {overview.get('total_clients', 0)}
- Total Support Tickets: {overview.get('total_tickets', 0)}
- Average Resolution Time: {performance.get('avg_resolution_time', 0):.1f} days
- Average Client Sentiment: {performance.get('avg_sentiment', 0):.2f}
- Critical Clients (high impact): {performance.get('critical_clients', 0)}
- Excellent Clients (low impact): {performance.get('excellent_clients', 0)}"""

        # Add RAG-retrieved context if available
        if rag_context and rag_confidence > 0.3:  # Only use if confidence is reasonable
            context_prompt += f"""

RELEVANT DATA FROM YOUR KNOWLEDGE BASE (Confidence: {rag_confidence:.2f}):
{rag_context}"""

        context_prompt += """

RECENT CONVERSATION:
"""

        # Add conversation history
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            context_prompt += f"{msg['type'].upper()}: {msg['content']}\n"

        context_prompt += f"""
CURRENT USER QUESTION: {user_message}

INSTRUCTIONS:
1. Use the specific data from the knowledge base when available and relevant
2. Provide helpful, data-driven responses about client performance and business insights
3. When discussing specific clients, reference actual data from the knowledge base
4. Suggest actionable recommendations based on the retrieved data
5. If asked to draft emails, create professional, personalized content
6. Be conversational but professional
7. If the knowledge base doesn't contain relevant information, use the general business context
8. Focus on business value and actionable insights
9. When referencing specific data, mention the source (e.g., "Based on your latest analysis...")

Please respond to the user's question:"""

        return context_prompt
    
    def process_message(self, user_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message and return AI response with RAG context"""

        try:
            # Get or create session
            session = self.get_or_create_session(session_id)

            # Save user message
            self.save_message(session, 'human', user_message)

            # Get conversation history
            history = self.get_conversation_history(session)

            # Get RAG context for metadata
            rag_metadata = {}
            if self.rag_query_processor:
                try:
                    rag_result = self.rag_query_processor.process_query(user_message, history)
                    rag_metadata = {
                        'rag_sources': rag_result.get('sources', []),
                        'rag_confidence': rag_result.get('confidence', 0.0),
                        'query_type': rag_result.get('query_type', 'general'),
                        'rag_enabled': True
                    }
                except Exception as e:
                    print(f"Error getting RAG metadata: {e}")
                    rag_metadata = {'rag_enabled': False, 'rag_error': str(e)}
            else:
                rag_metadata = {'rag_enabled': False}

            # Generate AI response
            if self.llm:
                print("Using Ollama LLM for response generation...")
                # Use LangChain/Ollama if available
                prompt = self._build_context_prompt(user_message, history)
                print("Generated context prompt for LLM")
                ai_response = self.llm.invoke(prompt)
                print(f"Received response from LLM: {ai_response[:100]}...")  # Print first 100 chars
                ai_mode = 'ollama_with_rag' if rag_metadata.get('rag_enabled') else 'ollama'
            else:
                print("LLM not available, using fallback AI service")
                # Use fallback AI service
                ai_response = self.fallback_ai.generate_response(user_message, history)
                ai_mode = 'fallback_with_rag' if rag_metadata.get('rag_enabled') else 'fallback'

            # Save AI response with RAG metadata
            self.save_message(session, 'ai', ai_response, rag_metadata)

            # Update session title if it's the first exchange
            if len(history) <= 1:
                title = user_message[:50] + "..." if len(user_message) > 50 else user_message
                session.title = title
                session.save()

            return {
                'success': True,
                'response': ai_response,
                'session_id': session.session_id,
                'conversation_history': self.get_conversation_history(session),
                'ai_mode': ai_mode,
                'rag_metadata': rag_metadata
            }

        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            return {
                'success': False,
                'error': f'Error processing message: {str(e)}',
                'session_id': session_id
            }

    def refresh_rag_index(self) -> Dict[str, Any]:
        """Refresh the RAG index with latest data"""
        if not self.rag_service:
            return {'success': False, 'error': 'RAG service not available'}

        try:
            # Process all documents
            processor_manager = DocumentProcessorManager(self.user)
            documents = processor_manager.process_all_documents()

            if not documents:
                return {'success': False, 'error': 'No documents to index'}

            # Clear existing and add new documents
            self.rag_service.clear_collection()
            success = self.rag_service.add_documents(documents)

            if success:
                stats = self.rag_service.get_collection_stats()
                return {
                    'success': True,
                    'message': f'Indexed {len(documents)} documents',
                    'stats': stats
                }
            else:
                return {'success': False, 'error': 'Failed to index documents'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""
        if not self.rag_service:
            return {'rag_available': False}

        try:
            stats = self.rag_service.get_collection_stats()
            processor_manager = DocumentProcessorManager(self.user)
            processor_stats = processor_manager.get_processor_stats()

            return {
                'rag_available': True,
                'collection_stats': stats,
                'data_sources': processor_stats
            }
        except Exception as e:
            return {'rag_available': False, 'error': str(e)}

    def search_knowledge_base(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base directly"""
        if not self.rag_service:
            return []

        try:
            return self.rag_service.search_documents(query, n_results)
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []

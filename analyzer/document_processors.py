"""
Document processors for converting business data into RAG-compatible documents
Handles client data, survey data, team metrics, and business insights
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from django.contrib.auth.models import User
from .models import AnalysisResult, SurveyAnalysis, ClientNote, JiraFile, SurveyFile, SurveyResponse

logger = logging.getLogger(__name__)


class BaseDocumentProcessor:
    """Base class for document processors"""
    
    def __init__(self, user: User):
        self.user = user
    
    def process(self) -> List[Dict[str, Any]]:
        """Process data and return list of documents for RAG indexing"""
        raise NotImplementedError
    
    def _create_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standardized document structure"""
        return {
            'id': doc_id,
            'content': content,
            'metadata': {
                **metadata,
                'user_id': self.user.id,
                'processed_at': datetime.now().isoformat()
            }
        }


class ClientDataProcessor(BaseDocumentProcessor):
    """Processes client analysis data from JIRA files with enhanced chunking"""

    def process(self) -> List[Dict[str, Any]]:
        """Process client analysis results into documents with improved chunking"""
        documents = []

        try:
            # Get all analysis results for the user - FIXED: More inclusive query
            analysis_results = AnalysisResult.objects.filter(
                jira_file__user=self.user
            ).select_related('jira_file').order_by('-created_at')

            logger.info(f"Found {analysis_results.count()} analysis results for user {self.user.username}")

            for analysis in analysis_results:
                # ENHANCED: Process overall analysis summary with more context
                summary_doc = self._create_analysis_summary_document(analysis)
                if summary_doc:
                    documents.append(summary_doc)

                # ENHANCED: Process individual client metrics with larger chunks
                client_docs = self._create_enhanced_client_documents(analysis)
                documents.extend(client_docs)

                # ENHANCED: Process client overview aggregated data
                overview_doc = self._create_client_overview_document(analysis)
                if overview_doc:
                    documents.append(overview_doc)

                # Process actionable insights
                insights_doc = self._create_insights_document(analysis)
                if insights_doc:
                    documents.append(insights_doc)

            logger.info(f"Processed {len(documents)} client data documents")
            return documents

        except Exception as e:
            logger.error(f"Error processing client data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _create_analysis_summary_document(self, analysis: AnalysisResult) -> Optional[Dict[str, Any]]:
        """Create a summary document for the analysis"""
        try:
            content_parts = []
            
            # Basic metrics
            content_parts.append(f"Analysis Summary for {analysis.jira_file.get_filename()}")
            content_parts.append(f"Total Issues: {analysis.issue_count}")
            
            # Ticket types
            if analysis.ticket_types:
                content_parts.append("Ticket Types Distribution:")
                for ticket_type, count in analysis.ticket_types.items():
                    content_parts.append(f"  - {ticket_type}: {count}")
            
            # Priority distribution
            if analysis.priority_distribution:
                content_parts.append("Priority Distribution:")
                for priority, count in analysis.priority_distribution.items():
                    content_parts.append(f"  - {priority}: {count}")
            
            # Status distribution
            if analysis.status_distribution:
                content_parts.append("Status Distribution:")
                for status, count in analysis.status_distribution.items():
                    content_parts.append(f"  - {status}: {count}")
            
            # Common themes
            if analysis.common_themes:
                content_parts.append("Common Themes:")
                for theme, frequency in analysis.common_themes.items():
                    content_parts.append(f"  - {theme}: {frequency}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                'doc_type': 'analysis_summary',
                'source': f'Analysis_{analysis.id}',
                'file_name': analysis.jira_file.get_filename(),
                'date': analysis.created_at.strftime('%Y-%m-%d'),
                'issue_count': analysis.issue_count
            }
            
            return self._create_document(
                doc_id=f"analysis_summary_{analysis.id}",
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating analysis summary document: {e}")
            return None
    
    def _create_enhanced_client_documents(self, analysis: AnalysisResult) -> List[Dict[str, Any]]:
        """Create enhanced individual documents for each client with larger chunks and better metadata"""
        documents = []

        try:
            if not analysis.client_metrics:
                logger.warning(f"No client metrics found for analysis {analysis.id}")
                return documents

            logger.info(f"Processing {len(analysis.client_metrics)} clients for analysis {analysis.id}")

            for client_name, metrics in analysis.client_metrics.items():
                # ENHANCED: Create comprehensive client overview chunks with larger context
                content_parts = []
                content_parts.append(f"=== CLIENT OVERVIEW: {client_name} ===")
                content_parts.append(f"Analysis Date: {analysis.created_at.strftime('%Y-%m-%d')}")
                content_parts.append(f"Source File: {analysis.jira_file.get_filename()}")
                content_parts.append(f"Data Source: Dashboard Analysis Results")
                content_parts.append("")

                # ENHANCED: Extract key client overview metrics that appear on the client overview page
                sentiment = metrics.get('sentiment', 0)
                tickets = metrics.get('Tickets', 0)
                resolution_time = metrics.get('Avg_Resolution_Time_Days', 0)
                client_impact = metrics.get('Client_Impact', 0) or metrics.get('Customer_Experience_Score', 0)
                priority_impact = metrics.get('Priority_Impact', 0)
                issue_type_impact = metrics.get('Issue_Type_Impact', 0)

                # ENHANCED: Create rich, searchable content that matches client overview page data
                content_parts.append("CLIENT OVERVIEW SUMMARY:")
                content_parts.append(f"This client overview shows performance metrics for {client_name} extracted from JIRA analysis.")
                content_parts.append(f"The client has {tickets} total tickets with an average resolution time of {resolution_time:.1f} days.")
                content_parts.append("")

                # Sentiment analysis section
                content_parts.append("SENTIMENT ANALYSIS:")
                sentiment_status = self._get_sentiment_status(sentiment)
                content_parts.append(f"  • Sentiment Score: {sentiment:.3f}")
                content_parts.append(f"  • Sentiment Status: {sentiment_status}")
                content_parts.append(f"  • Client satisfaction level based on ticket analysis")
                content_parts.append("")

                # Performance metrics section
                content_parts.append("PERFORMANCE METRICS:")
                content_parts.append(f"  • Total Tickets: {tickets}")
                content_parts.append(f"  • Average Resolution Time: {resolution_time:.1f} days")
                content_parts.append(f"  • Client Impact Score: {client_impact:.3f}")
                content_parts.append(f"  • Priority Impact: {priority_impact:.3f}")
                content_parts.append(f"  • Issue Type Impact: {issue_type_impact:.3f}")
                content_parts.append("")

                # Client categorization
                content_parts.append("CLIENT CATEGORIZATION:")
                if client_impact > 0.7:
                    content_parts.append(f"  • Category: Critical Client (High Impact)")
                elif client_impact > 0.4:
                    content_parts.append(f"  • Category: Important Client (Medium Impact)")
                else:
                    content_parts.append(f"  • Category: Standard Client (Low Impact)")

                if resolution_time > 5:
                    content_parts.append(f"  • Resolution Performance: Needs Attention (>{resolution_time:.1f} days)")
                elif resolution_time > 2:
                    content_parts.append(f"  • Resolution Performance: Average ({resolution_time:.1f} days)")
                else:
                    content_parts.append(f"  • Resolution Performance: Excellent ({resolution_time:.1f} days)")
                content_parts.append("")

                # Add all other metrics
                content_parts.append("DETAILED METRICS:")
                for metric_name, value in metrics.items():
                    if metric_name not in ['sentiment', 'Tickets', 'Avg_Resolution_Time_Days', 'Client_Impact', 'Customer_Experience_Score', 'Priority_Impact', 'Issue_Type_Impact']:
                        formatted_value = self._format_metric_value(value)
                        content_parts.append(f"  • {metric_name}: {formatted_value}")

                content = "\n".join(content_parts)

                # ENHANCED: Comprehensive metadata for better filtering and retrieval
                metadata = {
                    'doc_type': 'client_overview',
                    'source_type': 'client_analysis',
                    'source': f'Client_{client_name}',
                    'client_name': client_name,
                    'file_name': analysis.jira_file.get_filename(),
                    'date': analysis.created_at.strftime('%Y-%m-%d'),
                    'analysis_id': analysis.id,
                    'chunk_type': 'individual_client',
                    # Add key metrics to metadata for filtering
                    'sentiment_score': sentiment,
                    'sentiment_status': sentiment_status,
                    'total_tickets': tickets,
                    'resolution_time_days': resolution_time,
                    'client_impact_score': client_impact,
                    'priority_impact': priority_impact,
                    'issue_type_impact': issue_type_impact
                }

                # Add all numeric metrics to metadata
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        normalized_name = metric_name.lower().replace(' ', '_').replace('-', '_')
                        metadata[f'metric_{normalized_name}'] = value

                documents.append(self._create_document(
                    doc_id=f"client_overview_{analysis.id}_{client_name.replace(' ', '_').replace('/', '_')}",
                    content=content,
                    metadata=metadata
                ))

            return documents

        except Exception as e:
            logger.error(f"Error creating enhanced client documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _get_sentiment_status(self, sentiment: float) -> str:
        """Convert sentiment score to human-readable status"""
        if sentiment > 0.3:
            return "Positive"
        elif sentiment >= 0:
            return "Neutral"
        elif sentiment >= -0.2:
            return "Slightly Negative"
        elif sentiment >= -0.5:
            return "Negative"
        elif sentiment >= -0.8:
            return "Very Negative"
        else:
            return "Extremely Negative"

    def _format_metric_value(self, value) -> str:
        """Format metric values for better readability"""
        if isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, list):
            return ', '.join(map(str, value))
        else:
            return str(value)

    def _create_client_overview_document(self, analysis: AnalysisResult) -> Optional[Dict[str, Any]]:
        """Create aggregated client overview document for better retrieval"""
        try:
            if not analysis.client_metrics:
                return None

            content_parts = []
            content_parts.append(f"=== CLIENT OVERVIEW SUMMARY ===")
            content_parts.append(f"Analysis Date: {analysis.created_at.strftime('%Y-%m-%d')}")
            content_parts.append(f"Source File: {analysis.jira_file.get_filename()}")
            content_parts.append(f"Total Clients Analyzed: {len(analysis.client_metrics)}")
            content_parts.append("")

            # Aggregate metrics across all clients
            total_tickets = sum(metrics.get('Tickets', 0) for metrics in analysis.client_metrics.values())
            avg_resolution_time = sum(metrics.get('Avg_Resolution_Time_Days', 0) for metrics in analysis.client_metrics.values()) / len(analysis.client_metrics)
            avg_sentiment = sum(metrics.get('sentiment', 0) for metrics in analysis.client_metrics.values()) / len(analysis.client_metrics)

            content_parts.append("AGGREGATE METRICS:")
            content_parts.append(f"  • Total Tickets Across All Clients: {total_tickets}")
            content_parts.append(f"  • Average Resolution Time: {avg_resolution_time:.2f} days")
            content_parts.append(f"  • Average Client Sentiment: {avg_sentiment:.2f}")
            content_parts.append("")

            content_parts.append("CLIENT LIST:")
            for client_name, metrics in analysis.client_metrics.items():
                tickets = metrics.get('Tickets', 0)
                resolution_time = metrics.get('Avg_Resolution_Time_Days', 0)
                sentiment = metrics.get('sentiment', 0)
                content_parts.append(f"  • {client_name}: {tickets} tickets, {resolution_time:.1f}d avg resolution, {sentiment:.2f} sentiment")

            content = "\n".join(content_parts)

            metadata = {
                'doc_type': 'client_overview_summary',
                'source_type': 'aggregated_analysis',
                'source': f'Overview_{analysis.id}',
                'file_name': analysis.jira_file.get_filename(),
                'date': analysis.created_at.strftime('%Y-%m-%d'),
                'analysis_id': analysis.id,
                'chunk_type': 'overview_summary',
                'total_clients': len(analysis.client_metrics),
                'total_tickets': total_tickets,
                'avg_resolution_time': avg_resolution_time,
                'avg_sentiment': avg_sentiment
            }

            return self._create_document(
                doc_id=f"client_overview_summary_{analysis.id}",
                content=content,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error creating client overview document: {e}")
            return None

    def _create_insights_document(self, analysis: AnalysisResult) -> Optional[Dict[str, Any]]:
        """Create document for actionable insights"""
        try:
            if not analysis.actionable_insights:
                return None
            
            content_parts = []
            content_parts.append(f"Actionable Insights from {analysis.jira_file.get_filename()}")
            
            for insight in analysis.actionable_insights:
                if isinstance(insight, dict):
                    content_parts.append(f"- {insight.get('insight', str(insight))}")
                else:
                    content_parts.append(f"- {insight}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                'doc_type': 'business_insights',
                'source': f'Insights_{analysis.id}',
                'file_name': analysis.jira_file.get_filename(),
                'date': analysis.created_at.strftime('%Y-%m-%d'),
                'insight_count': len(analysis.actionable_insights)
            }
            
            return self._create_document(
                doc_id=f"insights_{analysis.id}",
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating insights document: {e}")
            return None


class SurveyDataProcessor(BaseDocumentProcessor):
    """Processes team survey data and analysis"""
    
    def process(self) -> List[Dict[str, Any]]:
        """Process survey analysis results into documents"""
        documents = []
        
        try:
            # Get all survey analyses for the user
            survey_analyses = SurveyAnalysis.objects.filter(
                survey_file__user=self.user,
                survey_file__processed=True
            ).select_related('survey_file').order_by('-created_at')
            
            for analysis in survey_analyses:
                # Process overall survey summary
                summary_doc = self._create_survey_summary_document(analysis)
                if summary_doc:
                    documents.append(summary_doc)
                
                # Process KPI metrics
                kpi_doc = self._create_kpi_document(analysis)
                if kpi_doc:
                    documents.append(kpi_doc)
                
                # Process feedback themes
                feedback_doc = self._create_feedback_document(analysis)
                if feedback_doc:
                    documents.append(feedback_doc)
            
            logger.info(f"Processed {len(documents)} survey data documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing survey data: {e}")
            return []
    
    def _create_survey_summary_document(self, analysis: SurveyAnalysis) -> Optional[Dict[str, Any]]:
        """Create summary document for survey analysis"""
        try:
            content_parts = []
            content_parts.append(f"Team Survey Analysis: {analysis.survey_file.get_filename()}")
            content_parts.append(f"Total Responses: {analysis.total_responses}")
            content_parts.append(f"Valid Responses: {analysis.valid_responses}")
            
            if analysis.overall_satisfaction:
                content_parts.append(f"Overall Satisfaction: {analysis.overall_satisfaction:.2f}/5.0")
            
            # Role distribution
            if analysis.role_distribution:
                content_parts.append("Role Distribution:")
                for role, count in analysis.role_distribution.items():
                    content_parts.append(f"  - {role}: {count}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                'doc_type': 'survey_summary',
                'source': f'Survey_{analysis.id}',
                'file_name': analysis.survey_file.get_filename(),
                'date': analysis.created_at.strftime('%Y-%m-%d'),
                'total_responses': analysis.total_responses,
                'overall_satisfaction': analysis.overall_satisfaction
            }
            
            return self._create_document(
                doc_id=f"survey_summary_{analysis.id}",
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating survey summary document: {e}")
            return None
    
    def _create_kpi_document(self, analysis: SurveyAnalysis) -> Optional[Dict[str, Any]]:
        """Create document for KPI metrics"""
        try:
            content_parts = []
            content_parts.append(f"Team KPI Metrics from {analysis.survey_file.get_filename()}")
            
            kpi_metrics = [
                ('Psychological Safety', analysis.avg_psychological_safety),
                ('Work Environment', analysis.avg_work_environment),
                ('Client Service', analysis.avg_client_service),
                ('Team Collaboration', analysis.avg_team_collaboration)
            ]
            
            for kpi_name, value in kpi_metrics:
                if value is not None:
                    content_parts.append(f"{kpi_name}: {value:.2f}/5.0")
            
            # Question averages
            if analysis.question_averages:
                content_parts.append("Detailed Question Averages:")
                for question, avg in analysis.question_averages.items():
                    content_parts.append(f"  - {question}: {avg:.2f}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                'doc_type': 'team_kpis',
                'source': f'KPIs_{analysis.id}',
                'file_name': analysis.survey_file.get_filename(),
                'date': analysis.created_at.strftime('%Y-%m-%d'),
                'psychological_safety': analysis.avg_psychological_safety,
                'work_environment': analysis.avg_work_environment,
                'client_service': analysis.avg_client_service,
                'team_collaboration': analysis.avg_team_collaboration
            }
            
            return self._create_document(
                doc_id=f"kpis_{analysis.id}",
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating KPI document: {e}")
            return None
    
    def _create_feedback_document(self, analysis: SurveyAnalysis) -> Optional[Dict[str, Any]]:
        """Create document for feedback themes"""
        try:
            if not analysis.feedback_themes or analysis.feedback_count == 0:
                return None
            
            content_parts = []
            content_parts.append(f"Team Feedback Themes from {analysis.survey_file.get_filename()}")
            content_parts.append(f"Total Feedback Responses: {analysis.feedback_count}")
            
            for theme in analysis.feedback_themes:
                if isinstance(theme, dict):
                    theme_text = theme.get('theme', str(theme))
                    content_parts.append(f"- {theme_text}")
                else:
                    content_parts.append(f"- {theme}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                'doc_type': 'team_feedback',
                'source': f'Feedback_{analysis.id}',
                'file_name': analysis.survey_file.get_filename(),
                'date': analysis.created_at.strftime('%Y-%m-%d'),
                'feedback_count': analysis.feedback_count,
                'theme_count': len(analysis.feedback_themes)
            }
            
            return self._create_document(
                doc_id=f"feedback_{analysis.id}",
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating feedback document: {e}")
            return None


class ClientNotesProcessor(BaseDocumentProcessor):
    """Processes client notes and qualitative data"""
    
    def process(self) -> List[Dict[str, Any]]:
        """Process client notes into documents"""
        documents = []
        
        try:
            # Get all client notes for the user
            client_notes = ClientNote.objects.filter(user=self.user).order_by('-updated_at')
            
            for note in client_notes:
                doc = self._create_note_document(note)
                if doc:
                    documents.append(doc)
            
            logger.info(f"Processed {len(documents)} client note documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing client notes: {e}")
            return []
    
    def _create_note_document(self, note: ClientNote) -> Optional[Dict[str, Any]]:
        """Create document for a client note"""
        try:
            content = f"Client Note for {note.client_name}:\n{note.note_text}"
            
            metadata = {
                'doc_type': 'client_notes',
                'source': f'Note_{note.id}',
                'client_name': note.client_name,
                'date': note.updated_at.strftime('%Y-%m-%d'),
                'created_date': note.created_at.strftime('%Y-%m-%d')
            }
            
            return self._create_document(
                doc_id=f"note_{note.id}",
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating note document: {e}")
            return None


class DocumentProcessorManager:
    """Manages all document processors and coordinates indexing"""
    
    def __init__(self, user: User):
        self.user = user
        self.processors = [
            ClientDataProcessor(user),
            SurveyDataProcessor(user),
            ClientNotesProcessor(user)
        ]
    
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all data sources and return combined document list"""
        all_documents = []
        
        for processor in self.processors:
            try:
                documents = processor.process()
                all_documents.extend(documents)
                logger.info(f"Processed {len(documents)} documents from {processor.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error in {processor.__class__.__name__}: {e}")
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get statistics about available data for processing"""
        stats = {}
        
        try:
            # Client data stats
            analysis_count = AnalysisResult.objects.filter(
                jira_file__user=self.user,
                jira_file__processed=True
            ).count()
            stats['analysis_results'] = analysis_count
            
            # Survey data stats
            survey_count = SurveyAnalysis.objects.filter(
                survey_file__user=self.user,
                survey_file__processed=True
            ).count()
            stats['survey_analyses'] = survey_count
            
            # Client notes stats
            notes_count = ClientNote.objects.filter(user=self.user).count()
            stats['client_notes'] = notes_count
            
            stats['total_data_sources'] = analysis_count + survey_count + notes_count
            
        except Exception as e:
            logger.error(f"Error getting processor stats: {e}")
            stats['error'] = str(e)
        
        return stats

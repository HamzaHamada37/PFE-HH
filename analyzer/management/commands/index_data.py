"""
Django management command to index data for RAG functionality
Usage: python manage.py index_data [--user username] [--clear] [--stats]
"""

from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from analyzer.rag_service import RAGService
from analyzer.document_processors import DocumentProcessorManager
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Index business data for RAG functionality'

    def add_arguments(self, parser):
        parser.add_argument(
            '--user',
            type=str,
            help='Username to index data for (if not provided, indexes for all users)',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing index before adding new documents',
        )
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Show statistics about indexed data',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output',
        )
        parser.add_argument(
            '--historical',
            action='store_true',
            help='Index all historical data comprehensively',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-indexing even if data is already indexed',
        )

    def handle(self, *args, **options):
        if options['verbose']:
            logging.basicConfig(level=logging.INFO)
        
        try:
            if options['stats']:
                self.show_stats(options.get('user'))
                return
            
            if options['user']:
                # Index data for specific user
                try:
                    user = User.objects.get(username=options['user'])
                    self.index_user_data(user, options['clear'], options['verbose'],
                                       options['historical'], options['force'])
                except User.DoesNotExist:
                    raise CommandError(f'User "{options["user"]}" does not exist')
            else:
                # Index data for all users
                users = User.objects.all()
                self.stdout.write(f"Indexing data for {users.count()} users...")

                for user in users:
                    self.index_user_data(user, options['clear'], options['verbose'],
                                       options['historical'], options['force'])

                self.stdout.write(
                    self.style.SUCCESS(f'Successfully indexed data for all users')
                )
        
        except Exception as e:
            raise CommandError(f'Error during indexing: {e}')

    def index_user_data(self, user: User, clear_existing: bool = False, verbose: bool = False,
                       historical: bool = False, force: bool = False):
        """Index data for a specific user with comprehensive options"""
        try:
            if verbose:
                self.stdout.write(f"Processing data for user: {user.username}")
            
            # Initialize RAG service
            rag_service = RAGService(user)
            
            if not rag_service.is_available():
                self.stdout.write(
                    self.style.WARNING(
                        f'RAG service not available for user {user.username}. '
                        'Make sure ChromaDB and sentence-transformers are installed.'
                    )
                )
                return
            
            # Clear existing data if requested
            if clear_existing:
                if verbose:
                    self.stdout.write(f"Clearing existing index for {user.username}")
                rag_service.clear_collection()
            
            # Process documents with comprehensive historical data
            processor_manager = DocumentProcessorManager(user)

            if historical:
                # Comprehensive historical indexing
                documents = self._process_historical_data(user, processor_manager, verbose)
            else:
                # Standard document processing
                documents = processor_manager.process_all_documents()

            if not documents:
                self.stdout.write(
                    self.style.WARNING(f'No documents found for user {user.username}')
                )
                return
            
            # Add documents to vector store
            success = rag_service.add_documents(documents)
            
            if success:
                stats = rag_service.get_collection_stats()
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully indexed {len(documents)} documents for {user.username}. '
                        f'Total documents in collection: {stats.get("total_documents", "unknown")}'
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'Failed to index documents for {user.username}')
                )
            
            if verbose:
                # Show processor stats
                processor_stats = processor_manager.get_processor_stats()
                self.stdout.write(f"Data sources for {user.username}:")
                self.stdout.write(f"  - Analysis results: {processor_stats.get('analysis_results', 0)}")
                self.stdout.write(f"  - Survey analyses: {processor_stats.get('survey_analyses', 0)}")
                self.stdout.write(f"  - Client notes: {processor_stats.get('client_notes', 0)}")
                self.stdout.write(f"  - Total documents indexed: {len(documents)}")
        
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error indexing data for {user.username}: {e}')
            )
            if verbose:
                import traceback
                self.stdout.write(traceback.format_exc())

    def show_stats(self, username: str = None):
        """Show statistics about indexed data"""
        try:
            if username:
                # Stats for specific user
                try:
                    user = User.objects.get(username=username)
                    users = [user]
                except User.DoesNotExist:
                    raise CommandError(f'User "{username}" does not exist')
            else:
                # Stats for all users
                users = User.objects.all()
            
            self.stdout.write("RAG Index Statistics:")
            self.stdout.write("=" * 50)
            
            total_documents = 0
            total_users_with_data = 0
            
            for user in users:
                rag_service = RAGService(user)
                
                if not rag_service.is_available():
                    self.stdout.write(f"{user.username}: RAG service not available")
                    continue
                
                stats = rag_service.get_collection_stats()
                doc_count = stats.get('total_documents', 0)
                
                if doc_count > 0:
                    total_users_with_data += 1
                    total_documents += doc_count
                
                self.stdout.write(f"{user.username}: {doc_count} documents")
                
                # Show data source breakdown
                processor_manager = DocumentProcessorManager(user)
                processor_stats = processor_manager.get_processor_stats()
                
                if any(processor_stats.get(key, 0) > 0 for key in ['analysis_results', 'survey_analyses', 'client_notes']):
                    self.stdout.write(f"  Data sources:")
                    self.stdout.write(f"    - Analysis results: {processor_stats.get('analysis_results', 0)}")
                    self.stdout.write(f"    - Survey analyses: {processor_stats.get('survey_analyses', 0)}")
                    self.stdout.write(f"    - Client notes: {processor_stats.get('client_notes', 0)}")
            
            self.stdout.write("=" * 50)
            self.stdout.write(f"Total users with indexed data: {total_users_with_data}")
            self.stdout.write(f"Total documents indexed: {total_documents}")
            
        except Exception as e:
            raise CommandError(f'Error showing stats: {e}')

    def _process_historical_data(self, user: User, processor_manager, verbose: bool = False):
        """Process comprehensive historical data from all sources"""
        from analyzer.models import AnalysisResult, SurveyAnalysis, ClientNote
        from datetime import datetime

        all_documents = []

        if verbose:
            self.stdout.write(f"🔍 Processing comprehensive historical data for {user.username}")

        # Get detailed statistics
        analysis_results = AnalysisResult.objects.filter(
            jira_file__user=user,
            jira_file__processed=True
        ).select_related('jira_file').order_by('-created_at')

        survey_analyses = SurveyAnalysis.objects.filter(
            survey_file__user=user,
            survey_file__processed=True
        ).select_related('survey_file').order_by('-created_at')

        client_notes = ClientNote.objects.filter(user=user).order_by('-updated_at')

        if verbose:
            self.stdout.write(f"  📊 Found {analysis_results.count()} analysis results")
            self.stdout.write(f"  📋 Found {survey_analyses.count()} survey analyses")
            self.stdout.write(f"  📝 Found {client_notes.count()} client notes")

        # Process each data source with detailed tracking
        try:
            # Process client analysis data
            client_processor = processor_manager.processors[0]  # ClientDataProcessor
            client_docs = client_processor.process()
            all_documents.extend(client_docs)

            if verbose and client_docs:
                self.stdout.write(f"  ✅ Processed {len(client_docs)} client analysis documents")
                # Show sample of what was processed
                for analysis in analysis_results[:3]:  # Show first 3
                    self.stdout.write(f"    - {analysis.jira_file.get_filename()} ({analysis.issue_count} issues)")

            # Process survey data
            survey_processor = processor_manager.processors[1]  # SurveyDataProcessor
            survey_docs = survey_processor.process()
            all_documents.extend(survey_docs)

            if verbose and survey_docs:
                self.stdout.write(f"  ✅ Processed {len(survey_docs)} survey documents")
                # Show sample of what was processed
                for analysis in survey_analyses[:3]:  # Show first 3
                    self.stdout.write(f"    - {analysis.survey_file.get_filename()} ({analysis.total_responses} responses)")

            # Process client notes
            notes_processor = processor_manager.processors[2]  # ClientNotesProcessor
            notes_docs = notes_processor.process()
            all_documents.extend(notes_docs)

            if verbose and notes_docs:
                self.stdout.write(f"  ✅ Processed {len(notes_docs)} client note documents")
                # Show sample of what was processed
                for note in client_notes[:3]:  # Show first 3
                    self.stdout.write(f"    - Note for {note.client_name} (updated {note.updated_at.strftime('%Y-%m-%d')})")

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error processing historical data: {e}')
            )
            if verbose:
                import traceback
                self.stdout.write(traceback.format_exc())

        if verbose:
            self.stdout.write(f"  📦 Total historical documents processed: {len(all_documents)}")

            # Show document type breakdown
            doc_types = {}
            for doc in all_documents:
                doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            self.stdout.write("  📈 Document type breakdown:")
            for doc_type, count in doc_types.items():
                self.stdout.write(f"    - {doc_type}: {count}")

        return all_documents

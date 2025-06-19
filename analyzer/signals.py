"""
Django signals for automatic RAG indexing when data is created or updated
Ensures real-time synchronization between database changes and vector store
"""

import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils import timezone
from .models import AnalysisResult, SurveyAnalysis, ClientNote, JiraFile, SurveyFile

logger = logging.getLogger(__name__)

# Flag to prevent recursive signal calls during bulk operations
_indexing_in_progress = False


def get_rag_service(user):
    """Get RAG service instance for a user"""
    try:
        from .rag_service import RAGService
        return RAGService(user)
    except ImportError:
        logger.warning("RAG service not available")
        return None


def get_document_processor_manager(user):
    """Get document processor manager for a user"""
    try:
        from .document_processors import DocumentProcessorManager
        return DocumentProcessorManager(user)
    except ImportError:
        logger.warning("Document processor manager not available")
        return None


def index_user_data(user, incremental=True):
    """Index or re-index all data for a specific user"""
    global _indexing_in_progress
    
    if _indexing_in_progress:
        return
    
    try:
        _indexing_in_progress = True
        
        rag_service = get_rag_service(user)
        if not rag_service or not rag_service.is_available():
            logger.warning(f"RAG service not available for user {user.username}")
            return
        
        processor_manager = get_document_processor_manager(user)
        if not processor_manager:
            logger.warning(f"Document processor not available for user {user.username}")
            return
        
        # Process all documents
        documents = processor_manager.process_all_documents()
        
        if not documents:
            logger.info(f"No documents to index for user {user.username}")
            return
        
        if not incremental:
            # Clear existing collection for full re-index
            rag_service.clear_collection()
        
        # Add documents to vector store
        success = rag_service.add_documents(documents)
        
        if success:
            logger.info(f"Successfully indexed {len(documents)} documents for user {user.username}")
        else:
            logger.error(f"Failed to index documents for user {user.username}")
            
    except Exception as e:
        logger.error(f"Error indexing data for user {user.username}: {e}")
    finally:
        _indexing_in_progress = False


@receiver(post_save, sender=AnalysisResult)
def analysis_result_saved(sender, instance, created, **kwargs):
    """Auto-index when AnalysisResult is created or updated"""
    try:
        user = instance.jira_file.user
        logger.info(f"AnalysisResult {'created' if created else 'updated'} for user {user.username}")
        
        # Mark as indexed and update timestamp
        if not instance.rag_indexed or not created:
            instance.rag_indexed = True
            instance.rag_indexed_at = timezone.now()
            # Use update to avoid triggering the signal again
            AnalysisResult.objects.filter(id=instance.id).update(
                rag_indexed=True,
                rag_indexed_at=timezone.now()
            )
        
        # Trigger incremental indexing
        index_user_data(user, incremental=True)
        
    except Exception as e:
        logger.error(f"Error in analysis_result_saved signal: {e}")


@receiver(post_save, sender=SurveyAnalysis)
def survey_analysis_saved(sender, instance, created, **kwargs):
    """Auto-index when SurveyAnalysis is created or updated"""
    try:
        user = instance.survey_file.user
        logger.info(f"SurveyAnalysis {'created' if created else 'updated'} for user {user.username}")
        
        # Mark as indexed and update timestamp
        if not instance.rag_indexed or not created:
            instance.rag_indexed = True
            instance.rag_indexed_at = timezone.now()
            # Use update to avoid triggering the signal again
            SurveyAnalysis.objects.filter(id=instance.id).update(
                rag_indexed=True,
                rag_indexed_at=timezone.now()
            )
        
        # Trigger incremental indexing
        index_user_data(user, incremental=True)
        
    except Exception as e:
        logger.error(f"Error in survey_analysis_saved signal: {e}")


@receiver(post_save, sender=ClientNote)
def client_note_saved(sender, instance, created, **kwargs):
    """Auto-index when ClientNote is created or updated"""
    try:
        user = instance.user
        logger.info(f"ClientNote {'created' if created else 'updated'} for user {user.username}")
        
        # Trigger incremental indexing
        index_user_data(user, incremental=True)
        
    except Exception as e:
        logger.error(f"Error in client_note_saved signal: {e}")


@receiver(post_delete, sender=AnalysisResult)
def analysis_result_deleted(sender, instance, **kwargs):
    """Remove from index when AnalysisResult is deleted"""
    try:
        user = instance.jira_file.user
        logger.info(f"AnalysisResult deleted for user {user.username}")
        
        # Trigger full re-indexing to remove deleted documents
        index_user_data(user, incremental=False)
        
    except Exception as e:
        logger.error(f"Error in analysis_result_deleted signal: {e}")


@receiver(post_delete, sender=SurveyAnalysis)
def survey_analysis_deleted(sender, instance, **kwargs):
    """Remove from index when SurveyAnalysis is deleted"""
    try:
        user = instance.survey_file.user
        logger.info(f"SurveyAnalysis deleted for user {user.username}")
        
        # Trigger full re-indexing to remove deleted documents
        index_user_data(user, incremental=False)
        
    except Exception as e:
        logger.error(f"Error in survey_analysis_deleted signal: {e}")


@receiver(post_delete, sender=ClientNote)
def client_note_deleted(sender, instance, **kwargs):
    """Remove from index when ClientNote is deleted"""
    try:
        user = instance.user
        logger.info(f"ClientNote deleted for user {user.username}")
        
        # Trigger full re-indexing to remove deleted documents
        index_user_data(user, incremental=False)
        
    except Exception as e:
        logger.error(f"Error in client_note_deleted signal: {e}")


@receiver(post_save, sender=JiraFile)
def jira_file_processed(sender, instance, created, **kwargs):
    """Trigger indexing when JIRA file processing is completed"""
    try:
        # Only trigger when file is marked as processed
        if instance.processed and not created:
            user = instance.user
            logger.info(f"JIRA file processing completed for user {user.username}: {instance.get_filename()}")
            
            # The AnalysisResult signal will handle the actual indexing
            # This is just for logging and potential future enhancements
            
    except Exception as e:
        logger.error(f"Error in jira_file_processed signal: {e}")


@receiver(post_save, sender=SurveyFile)
def survey_file_processed(sender, instance, created, **kwargs):
    """Trigger indexing when Survey file processing is completed"""
    try:
        # Only trigger when file is marked as processed
        if instance.processed and not created:
            user = instance.user
            logger.info(f"Survey file processing completed for user {user.username}: {instance.get_filename()}")
            
            # The SurveyAnalysis signal will handle the actual indexing
            # This is just for logging and potential future enhancements
            
    except Exception as e:
        logger.error(f"Error in survey_file_processed signal: {e}")


def bulk_index_all_users():
    """Utility function to bulk index all users' data (for initial setup)"""
    from django.contrib.auth.models import User
    
    logger.info("Starting bulk indexing for all users")
    
    users = User.objects.all()
    for user in users:
        try:
            logger.info(f"Bulk indexing data for user: {user.username}")
            index_user_data(user, incremental=False)
        except Exception as e:
            logger.error(f"Error bulk indexing for user {user.username}: {e}")
    
    logger.info("Bulk indexing completed for all users")


def force_reindex_user(user):
    """Force a complete re-index for a specific user"""
    logger.info(f"Force re-indexing data for user: {user.username}")
    index_user_data(user, incremental=False)

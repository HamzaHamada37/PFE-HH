from django.db import models
from django.contrib.auth.models import User
import json
import uuid

class JiraFile(models.Model):
    FILE_TYPE_CHOICES = [
        ('csv', 'CSV'),
        ('excel', 'Excel'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='jira_files')
    file = models.FileField(upload_to='jira_files/')
    original_filename = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    analysis_date = models.DateField(null=True, blank=True, help_text='Date selected for analysis filtering and context')

    def __str__(self):
        return f"{self.original_filename or self.file.name} (Uploaded by {self.user.username})"

    def get_filename(self):
        """Return the original filename if available, otherwise extract from path"""
        if self.original_filename:
            return self.original_filename
        return self.file.name.split('/')[-1]

class AnalysisResult(models.Model):
    jira_file = models.ForeignKey(JiraFile, on_delete=models.CASCADE, related_name='analysis_results')
    issue_count = models.IntegerField(default=0)
    ticket_types = models.JSONField(default=dict)
    priority_distribution = models.JSONField(default=dict)
    status_distribution = models.JSONField(default=dict)
    common_themes = models.JSONField(default=dict)
    sentiment_analysis = models.JSONField(default=dict)
    client_metrics = models.JSONField(default=dict)
    actionable_insights = models.JSONField(default=list)
    theme_visualization = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    # RAG indexing metadata
    rag_indexed = models.BooleanField(default=False)
    rag_indexed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Analysis for {self.jira_file.file.name}"


class ClientNote(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='client_notes')
    client_name = models.CharField(max_length=255)
    note_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        unique_together = ['user', 'client_name']

    def __str__(self):
        return f"Note for {self.client_name} by {self.user.username}"

    def save(self, *args, **kwargs):
        # ClientNote model doesn't need special field initialization
        super().save(*args, **kwargs)


class ChatSession(models.Model):
    """Model to track chat sessions for conversation continuity"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    session_id = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=255, default="New Conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"Chat Session {self.session_id} - {self.user.username}"


class ChatMessage(models.Model):
    """Model to store individual chat messages"""
    MESSAGE_TYPES = [
        ('human', 'Human'),
        ('ai', 'AI'),
        ('system', 'System'),
    ]

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)  # Store additional context like data sources used
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."


class SurveyFile(models.Model):
    """Model to store uploaded survey files"""
    FILE_TYPE_CHOICES = [
        ('xlsx', 'Excel (.xlsx)'),
        ('xls', 'Excel (.xls)'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='survey_files')
    file = models.FileField(upload_to='survey_files/')
    original_filename = models.CharField(max_length=255, blank=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    survey_date = models.DateField(null=True, blank=True, help_text='Date when the survey was conducted')

    def __str__(self):
        return f"{self.original_filename or self.file.name} (Uploaded by {self.user.username})"

    def get_filename(self):
        """Return the original filename if available, otherwise extract from path"""
        if self.original_filename:
            return self.original_filename
        return self.file.name.split('/')[-1]


class SurveyResponse(models.Model):
    """Model to store individual survey responses"""
    survey_file = models.ForeignKey(SurveyFile, on_delete=models.CASCADE, related_name='responses')
    response_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)

    # Basic info
    role = models.CharField(max_length=100, blank=True)

    # Psychological Safety (Q1)
    q1_1_speaking_up = models.IntegerField(null=True, blank=True, help_text='Speaking up comfort level (1-5)')
    q1_2_mistakes_held_against = models.IntegerField(null=True, blank=True, help_text='Mistakes held against (1-5)')
    q1_3_respect_when_not_knowing = models.IntegerField(null=True, blank=True, help_text='Respect when not knowing (1-5)')

    # Work Environment (Q2)
    q2_1_workload_manageable = models.IntegerField(null=True, blank=True, help_text='Workload manageability (1-5)')
    q2_2_tools_and_resources = models.IntegerField(null=True, blank=True, help_text='Tools and resources adequacy (1-5)')
    q2_3_work_life_balance = models.IntegerField(null=True, blank=True, help_text='Work-life balance (1-5)')

    # Client Service (Q3)
    q3_1_understanding_clients = models.FloatField(null=True, blank=True, help_text='Understanding client needs (1-5)')
    q3_2_support_handling_clients = models.FloatField(null=True, blank=True, help_text='Support in handling clients (1-5)')
    q3_3_tools_for_client_service = models.FloatField(null=True, blank=True, help_text='Tools for client service (1-5)')

    # Team Collaboration (Q4)
    q4_1_help_responsiveness = models.IntegerField(null=True, blank=True, help_text='Help responsiveness (1-5)')
    q4_2_conflict_resolution = models.IntegerField(null=True, blank=True, help_text='Conflict resolution (1-5)')
    q4_3_sharing_updates = models.IntegerField(null=True, blank=True, help_text='Sharing updates (1-5)')

    # Open feedback
    open_feedback = models.TextField(blank=True, help_text='Open-ended feedback')

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Survey Response {self.response_id} - {self.role}"

    class Meta:
        ordering = ['-created_at']


class SurveyAnalysis(models.Model):
    """Model to store aggregated survey analysis results"""
    survey_file = models.OneToOneField(SurveyFile, on_delete=models.CASCADE, related_name='analysis')

    # Response counts
    total_responses = models.IntegerField(default=0)
    valid_responses = models.IntegerField(default=0)  # Responses without N/A values

    # KPI Averages (calculated from valid responses only)
    avg_psychological_safety = models.FloatField(null=True, blank=True)
    avg_work_environment = models.FloatField(null=True, blank=True)
    avg_client_service = models.FloatField(null=True, blank=True)
    avg_team_collaboration = models.FloatField(null=True, blank=True)
    overall_satisfaction = models.FloatField(null=True, blank=True)

    # Detailed metrics
    role_distribution = models.JSONField(default=dict)
    question_averages = models.JSONField(default=dict)  # Individual question averages
    satisfaction_distribution = models.JSONField(default=dict)  # Distribution of satisfaction levels

    # Qualitative analysis
    feedback_themes = models.JSONField(default=list)
    feedback_count = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # RAG indexing metadata
    rag_indexed = models.BooleanField(default=False)
    rag_indexed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Analysis for {self.survey_file.get_filename()}"
